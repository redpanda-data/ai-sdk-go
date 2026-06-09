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

package tool_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/tool"
)

func TestAwait_Validate_AcceptsAllowedPairs(t *testing.T) {
	t.Parallel()

	cases := []struct {
		reason tool.AwaitReason
		resume tool.ResumeMode
	}{
		{tool.AwaitReasonToolResult, tool.ResumeWithToolResponse},
		{tool.AwaitReasonToolResult, tool.ResumeWithReentry},
		{tool.AwaitReasonUserInput, tool.ResumeWithMessage},
		{tool.AwaitReasonApproval, tool.ResumeWithReentry},
		{tool.AwaitReasonElicitation, tool.ResumeWithReentry},
		{tool.AwaitReasonHandoff, tool.ResumeWithReentry},
	}

	for _, tc := range cases {
		t.Run(string(tc.reason)+"+"+string(tc.resume), func(t *testing.T) {
			t.Parallel()

			a := &tool.Await{Reason: tc.reason, Resume: tc.resume}
			require.NoError(t, a.Validate())
		})
	}
}

func TestAwait_Validate_RejectsForbiddenPairs(t *testing.T) {
	t.Parallel()

	cases := []struct {
		reason tool.AwaitReason
		resume tool.ResumeMode
	}{
		// user_input cannot resume via tool_response or reentry
		{tool.AwaitReasonUserInput, tool.ResumeWithToolResponse},
		{tool.AwaitReasonUserInput, tool.ResumeWithReentry},
		// approval cannot resume via tool_response or message
		{tool.AwaitReasonApproval, tool.ResumeWithToolResponse},
		{tool.AwaitReasonApproval, tool.ResumeWithMessage},
		// elicitation cannot resume via tool_response or message
		{tool.AwaitReasonElicitation, tool.ResumeWithToolResponse},
		{tool.AwaitReasonElicitation, tool.ResumeWithMessage},
		// handoff cannot resume via tool_response or message
		{tool.AwaitReasonHandoff, tool.ResumeWithToolResponse},
		{tool.AwaitReasonHandoff, tool.ResumeWithMessage},
		// tool_result cannot resume via message
		{tool.AwaitReasonToolResult, tool.ResumeWithMessage},
	}

	for _, tc := range cases {
		t.Run(string(tc.reason)+"+"+string(tc.resume), func(t *testing.T) {
			t.Parallel()

			a := &tool.Await{Reason: tc.reason, Resume: tc.resume}
			err := a.Validate()
			require.Error(t, err)
			assert.ErrorIs(t, err, tool.ErrAwaitInvalid)
		})
	}
}

func TestAwait_Validate_MissingFields(t *testing.T) {
	t.Parallel()

	t.Run("nil receiver is valid", func(t *testing.T) {
		t.Parallel()

		var a *tool.Await
		require.NoError(t, a.Validate())
	})

	t.Run("missing reason", func(t *testing.T) {
		t.Parallel()

		a := &tool.Await{Resume: tool.ResumeWithMessage}
		assert.ErrorIs(t, a.Validate(), tool.ErrAwaitReasonEmpty)
	})

	t.Run("missing resume", func(t *testing.T) {
		t.Parallel()

		a := &tool.Await{Reason: tool.AwaitReasonUserInput}
		assert.ErrorIs(t, a.Validate(), tool.ErrAwaitResumeEmpty)
	})

	t.Run("unknown reason", func(t *testing.T) {
		t.Parallel()

		a := &tool.Await{Reason: "made_up", Resume: tool.ResumeWithMessage}
		assert.ErrorIs(t, a.Validate(), tool.ErrAwaitInvalid)
	})
}

func TestAwaitOptions(t *testing.T) {
	t.Parallel()

	a := &tool.Await{Reason: tool.AwaitReasonUserInput, Resume: tool.ResumeWithMessage}
	tool.WithAwaitMessage("please confirm")(a)
	tool.WithCorrelationID("job-123")(a)
	tool.WithAwaitMetadata(map[string]any{"k": "v"})(a)

	assert.Equal(t, "please confirm", a.Message)
	assert.Equal(t, "job-123", a.CorrelationID)
	assert.Equal(t, "v", a.Metadata["k"])

	// Mutating the input map after WithAwaitMetadata must not leak.
	src := map[string]any{"k": "v"}
	tool.WithAwaitMetadata(src)(a)
	src["k"] = "mutated"

	assert.Equal(t, "v", a.Metadata["k"], "metadata should be cloned on set")
}
