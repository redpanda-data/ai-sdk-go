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

package bedrock

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/bedrock/bedrocktest"
)

// TestBedrockOmittedMaxTokensDefault_Integration is the live boundary check that
// establishes the fact the Claude default max_tokens injection depends on: that
// Converse, given no maxTokens for a Claude model, applies a small implicit
// output cap far below the model's advertised MaxOutputTokens — rather than the
// model maximum the published InferenceConfiguration reference describes. Unit
// tests cannot settle this: they assert config state after the injection, not
// what AWS does on the wire when the field is absent. This one requires
// authorized Bedrock credentials and is skipped without them.
//
// It runs the same long-output prompt twice against the same catalog Claude
// model and compares the observed output-token counts:
//
//   - Omitted: maxTokens forced back to nil so the request carries no maxTokens
//     (NewModel injects the default for Claude, so it is cleared here to observe
//     the raw provider behavior the default is meant to correct).
//   - Explicit: maxTokens set to the injected default so its ceiling is visible.
//
// The prompt asks for far more output than any cap allows, so a run that hits a
// ceiling stops with FinishReasonLength at roughly that ceiling. If the omitted
// run truncates below the explicit 16K run, the implicit default is the low cap
// the injection is meant to lift. If instead it runs to the model maximum, the
// injection lowers the effective ceiling and must be dropped — the assertions
// fail loudly and say so.
func TestBedrockOmittedMaxTokensDefault_Integration(t *testing.T) {
	t.Parallel()

	bedrocktest.SkipUnlessAWSCredentials(t)

	region := bedrocktest.TestRegion

	provider, err := NewProvider(context.Background(), WithRegion(region))
	require.NoError(t, err)

	// A prompt that demands far more output than any per-turn cap allows, so the
	// turn is cut off by the budget (FinishReasonLength) rather than finishing on
	// its own. That cutoff point is the effective max_tokens we want to observe.
	longOutputPrompt := "Write an extremely long, continuous fictional story of at " +
		"least 100,000 words. Do not summarize, do not stop early, do not ask " +
		"questions, and never wrap up — keep writing prose until you are cut off."

	// Omitted run: build via NewModel (which injects the Claude default), then
	// clear MaxTokens so the request carries no maxTokens at all. The request
	// mapper reads this same Config pointer at send time, so nil-ing it here is
	// what puts the field-absent case on the wire.
	omitted, err := provider.NewModel(bedrocktest.TestModelName)
	require.NoError(t, err)

	om, ok := omitted.(*Model)
	require.True(t, ok)
	require.NotNil(t, om.config.MaxTokens, "NewModel should inject the Claude default before we clear it")
	om.config.MaxTokens = nil

	outputCap := om.definition.Constraints.MaxOutputTokens
	require.Positive(t, outputCap, "catalog model must advertise an output cap for this check to be meaningful")

	omittedOut, omittedFinish := generateAndMeasure(t, omitted, longOutputPrompt)
	t.Logf("omitted maxTokens: output_tokens=%d finish=%q (model cap=%d, SDK default=%d)",
		omittedOut, omittedFinish, outputCap, defaultMaxTokens)

	// Explicit run: the injected default made visible, so we can compare ceilings.
	explicit, err := provider.NewModel(bedrocktest.TestModelName, WithMaxTokens(defaultMaxTokens))
	require.NoError(t, err)

	explicitOut, explicitFinish := generateAndMeasure(t, explicit, longOutputPrompt)
	t.Logf("explicit maxTokens=%d: output_tokens=%d finish=%q", defaultMaxTokens, explicitOut, explicitFinish)

	// Both runs must have been cut off by their budget; otherwise the prompt
	// finished on its own and neither number reflects a cap.
	require.Equal(t, llm.FinishReasonLength, omittedFinish,
		"prompt did not exhaust the omitted-maxTokens budget; cannot characterize the implicit default")
	require.Equal(t, llm.FinishReasonLength, explicitFinish,
		"prompt did not exhaust the explicit %d budget; cannot compare ceilings", defaultMaxTokens)

	// The explicit default should cap output near 16K — proof the SDK value
	// reaches the wire and is the operative ceiling when set.
	assert.Greater(t, explicitOut, 12000,
		"explicit maxTokens=%d should produce output near that ceiling; got %d", defaultMaxTokens, explicitOut)

	// The premise: omitting maxTokens truncates BELOW the injected default, so
	// the injection raises the ceiling rather than lowering it. If the implicit
	// default were the model maximum (as the published reference states), this
	// run would produce far more than 16K and the assertion fails — the signal
	// that the injection is harmful and PR must be reconsidered.
	assert.Less(t, omittedOut, defaultMaxTokens,
		"PREMISE CHECK: omitting maxTokens should truncate below the injected %d default; "+
			"got %d output tokens. If this meets or exceeds %d, Bedrock's implicit default is "+
			"the model maximum (%d), not a low cap — the injected default LOWERS the effective "+
			"ceiling and must be removed.", defaultMaxTokens, omittedOut, defaultMaxTokens, outputCap)
}

// generateAndMeasure runs one generation and returns the visible output-token
// count and finish reason.
func generateAndMeasure(t *testing.T, model llm.Model, prompt string) (int, llm.FinishReason) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()

	resp, err := model.Generate(ctx, &llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, llm.NewTextPart(prompt)),
		},
	})
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, resp.Usage, "provider must report token usage for this check")

	return resp.Usage.OutputTokens, resp.FinishReason
}
