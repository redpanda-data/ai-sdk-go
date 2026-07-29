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

package agenttool_test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic/anthropictest"
	"github.com/redpanda-data/ai-sdk-go/tool/agenttool"
)

// TestExecute_TruncationFlag_Integration proves end to end, against the real
// Anthropic API, that a genuinely truncated sub-agent turn surfaces as
// Result.Truncated. A real llmagent runs behind the agent-as-tool, and a tiny
// per-model output budget forces the model to stop at the output-token cap —
// exercising the full model → llmagent → agenttool finish-reason path that the
// mock-based unit tests cannot.
func TestExecute_TruncationFlag_Integration(t *testing.T) {
	t.Parallel()

	apiKey := anthropictest.GetAPIKeyOrSkipTest(t)

	provider, err := anthropic.NewProvider(apiKey)
	require.NoError(t, err)

	// A tiny output budget guarantees the long-essay prompt truncates.
	model, err := provider.NewModel(anthropictest.TestModelName, anthropic.WithMaxTokens(16))
	require.NoError(t, err)

	subAgent, err := llmagent.New("essayist", "You are a helpful assistant.", model)
	require.NoError(t, err)

	at := agenttool.New(subAgent)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	args, err := json.Marshal(map[string]string{
		"query": "Write a detailed 1000-word essay about the history of computing.",
	})
	require.NoError(t, err)

	raw, err := at.Execute(ctx, args)
	require.NoError(t, err)

	var output agenttool.Result
	require.NoError(t, json.Unmarshal(raw, &output))

	assert.Equal(t, true, output.Metadata["truncated"],
		"a 16-token cap on a long-essay prompt must surface as a truncated sub-agent result")
	assert.NotEmpty(t, output.Result,
		"the partial content the sub-agent produced before the cut must still be delivered")

	t.Logf("metadata=%v result=%q", output.Metadata, output.Result)
}
