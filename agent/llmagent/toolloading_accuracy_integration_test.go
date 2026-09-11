// Copyright 2026 Redpanda Data, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package llmagent

import (
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic/anthropictest"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// accuracyTask is one request with exactly one right answer.
type accuracyTask struct {
	prompt string
	group  string
	want   string
}

// accuracyTasks are phrased the way a Teams user would, deliberately without
// naming the tool: the point is whether the model can get from intent to the
// right tool through the manifest and tool_search.
var accuracyTasks = []accuracyTask{
	{
		prompt: "Please open a new ServiceNow ticket for record INC0099 - the laptop will not boot.",
		group:  "servicenow",
		want:   "servicenow__create_record",
	},
	{
		prompt: "Mark ServiceNow record INC0042 as finished and shut it.",
		group:  "servicenow",
		want:   "servicenow__close_record",
	},
	{
		prompt: "Put a note on Jira record PROJ-7 saying the fix is merged.",
		group:  "jira",
		want:   "jira__comment_record",
	},
	{
		prompt: "Hand Jira record PROJ-9 to the platform team.",
		group:  "jira",
		want:   "jira__assign_record",
	},
	{
		prompt: "Pull down the Confluence record RUNBOOK-1 so I can read it.",
		group:  "confluence",
		want:   "confluence__get_record",
	},
	{
		prompt: "Connect Datadog record MON-3 to its upstream record.",
		group:  "datadog",
		want:   "datadog__link_record",
	},
}

// TestToolLoadingAccuracyUnderDistractors is the accuracy half of the
// evaluation plan for local discovery: can the model reach the right tool, and
// does it get there in one search, as irrelevant tool groups grow? singleShot
// measures the local select: protocol, not native hosted search.
func TestToolLoadingAccuracyUnderDistractors_Integration(t *testing.T) {
	t.Parallel()

	if os.Getenv("RUN_ACCURACY_EVAL") != "1" {
		t.Skip("measurement harness, not a regression test: set RUN_ACCURACY_EVAL=1 to run it")
	}

	apiKey := anthropictest.GetAPIKeyOrSkipTest(t)

	provider, err := anthropic.NewProvider(apiKey,
		anthropic.WithCaching(),
		anthropic.WithTimeout(3*time.Minute),
	)
	require.NoError(t, err)

	// Distractor levels: how many groups beyond the task's own are present.
	// 0 distractors is 10 tools; 5 is the full 60.
	for _, distractors := range []int{0, 2, 4, 5} {
		for _, arm := range []struct {
			name string
			lazy bool
		}{{"local-lazy", true}, {"always-on", false}} {
			t.Run(fmt.Sprintf("distractors=%d/%s", distractors, arm.name), func(t *testing.T) {
				t.Parallel()

				var (
					correct        int
					singleShot     int
					totalModelCall int
					totalToolCall  int
					attempted      int
				)

				for _, task := range accuracyTasks {
					model, err := provider.NewModel(anthropictest.TestModelName)
					require.NoError(t, err)

					registry := buildAccuracyRegistry(t, groupsFor(task.group, distractors), arm.lazy)

					// Keep the local search metrics comparable when the model gains
					// native search support. Native behavior has separate live tests.
					opts := []Option{WithTools(registry), WithMaxTurns(6), WithToolLoadingConfig(ToolLoadingConfig{ForceLocal: true})}

					ag, err := New("service-desk",
						"You are an internal service desk agent. Use the available tools to act on the "+
							"user's request. Do not ask for confirmation.",
						model, opts...)
					require.NoError(t, err)

					sess := &session.State{
						ID: fmt.Sprintf("acc-%d-%s-%s", distractors, arm.name, task.want),
						Messages: []llm.Message{
							llm.NewMessage(llm.RoleUser, llm.NewTextPart(task.prompt)),
						},
					}

					run := observeRun(t, ag.Run(t.Context(), agent.NewInvocationMetadata(sess, agent.Info{})))
					outcome := run.scoreAgainst(task.want)

					attempted++

					if outcome.succeeded {
						correct++
						totalModelCall += outcome.modelCalls
						totalToolCall += outcome.toolCalls
					}

					if outcome.succeeded && outcome.singleShot {
						singleShot++
					}

					t.Logf("d=%d arm=%-9s task=%-28s ok=%-5t single_shot=%-5t model_calls=%d tools=%v",
						distractors, arm.name, task.want, outcome.succeeded, outcome.singleShot,
						outcome.modelCalls, run.toolNames())
				}

				meanModel, meanTool := 0.0, 0.0
				if correct > 0 {
					meanModel = float64(totalModelCall) / float64(correct)
					meanTool = float64(totalToolCall) / float64(correct)
				}

				t.Logf("SUMMARY d=%d arm=%s correct=%d/%d single_shot=%d/%d "+
					"mean_model_calls_to_correct=%.2f mean_tool_calls_to_correct=%.2f",
					distractors, arm.name, correct, attempted, singleShot, attempted,
					meanModel, meanTool)

				// Deliberately loose: this is a measurement harness first, and
				// provider behaviour is not deterministic - the same configuration
				// has scored 4/6 at one level and 6/6 at the three others in one
				// run. It fails only on a real regression: the model unable to
				// reach its tool at all.
				assert.GreaterOrEqual(t, correct, attempted-2,
					"at most two tasks may miss their tool; see the per-task logs above")
			})
		}
	}
}

// runObservation is one conversation's event stream, reduced to what scoring
// needs. Model calls and tool calls are counted separately: latency tracks
// model calls, while one model call can carry several tool calls.
type runObservation struct {
	modelCalls int
	tools      []toolAttempt
}

type toolAttempt struct {
	name        string
	arguments   string
	modelCalls  int
	failed      bool
	sawResponse bool
}

type runOutcome struct {
	succeeded  bool
	singleShot bool
	modelCalls int
	toolCalls  int
}

// observeRun drains an agent run, pairing each tool request with its result so
// a call that came back as an error is not counted as a success.
func observeRun(tb testing.TB, iter func(func(agent.Event, error) bool)) runObservation {
	tb.Helper()

	var obs runObservation

	byID := map[string]int{}

	for evt, err := range iter {
		require.NoError(tb, err, "unexpected error in event stream")

		switch typed := evt.(type) {
		case agent.MessageEvent:
			obs.modelCalls++
		case agent.ToolRequestEvent:
			byID[typed.Request.ID] = len(obs.tools)
			obs.tools = append(obs.tools, toolAttempt{
				name:       typed.Request.Name,
				arguments:  string(typed.Request.Arguments),
				modelCalls: obs.modelCalls,
			})
		case agent.ToolResponseEvent:
			idx, ok := byID[typed.Response.ID]
			if !ok {
				continue
			}

			obs.tools[idx].sawResponse = true
			obs.tools[idx].failed = typed.Response.IsError
		}
	}

	return obs
}

// scoreAgainst reports the first successful call of want, and how much it cost
// to get there.
func (o runObservation) scoreAgainst(want string) runOutcome {
	var firstSearch *toolAttempt

	for i := range o.tools {
		attempt := o.tools[i]

		if attempt.name == toolSearchName && firstSearch == nil {
			firstSearch = &o.tools[i]
		}

		// A request alone is not a success: the self-heal path answers a call on
		// an unloaded tool with an error, and scoring on the name would count
		// that failed attempt as the model having done the job.
		if attempt.name != want || attempt.failed || !attempt.sawResponse {
			continue
		}

		return runOutcome{
			succeeded:  true,
			singleShot: firstSearch != nil && searchUsedSelectFor(firstSearch.arguments, want),
			modelCalls: attempt.modelCalls,
			toolCalls:  i + 1,
		}
	}

	return runOutcome{}
}

func (o runObservation) toolNames() []string {
	names := make([]string, len(o.tools))
	for i, attempt := range o.tools {
		names[i] = attempt.name
		if attempt.failed {
			names[i] += "!"
		}
	}

	return names
}

// groupsFor returns the task's own group plus n distractor groups, always in
// the same order so a rerun measures the same thing.
func groupsFor(own string, distractors int) []string {
	groups := []string{own}

	for _, candidate := range measureGroups {
		if len(groups) > distractors {
			break
		}

		if candidate != own {
			groups = append(groups, candidate)
		}
	}

	slices.Sort(groups)

	return groups
}

// buildAccuracyRegistry is buildMeasureRegistry restricted to the named groups.
// With lazy set, every tool is deferred except each group's search entry point;
// without it nothing is deferred, which is the always-on control arm.
func buildAccuracyRegistry(tb testing.TB, groups []string, lazy bool) tool.Registry {
	tb.Helper()

	registry := tool.NewRegistry(tool.RegistryConfig{})

	for _, def := range buildMeasureRegistry(tb, lazy).List() {
		if def.Group.Name != "" && !slices.Contains(groups, def.Group.Name) {
			continue
		}

		var opts []tool.Option
		if def.Group.Name != "" {
			opts = append(opts, tool.WithGroup(def.Group))
		}

		if def.Deferred {
			opts = append(opts, tool.WithDeferred())
		}

		require.NoError(tb, registry.Register(&stubTool{def: def}, opts...))
	}

	return registry
}

// searchUsedSelectFor reports whether one tool_search call used the select:
// form and named the wanted tool - the "found it from the manifest in one shot"
// signal.
func searchUsedSelectFor(arguments, want string) bool {
	args := strings.ToLower(arguments)

	return strings.Contains(args, "select:") && strings.Contains(args, strings.ToLower(want))
}
