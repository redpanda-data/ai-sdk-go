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
	"context"
	"errors"
	"iter"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

type nonStreamingModel struct {
	generateCalls       int
	generateEventsCalls int
}

func (*nonStreamingModel) Name() string     { return "nonstreaming" }
func (*nonStreamingModel) Provider() string { return "test" }
func (*nonStreamingModel) Capabilities() llm.ModelCapabilities {
	return llm.ModelCapabilities{Streaming: false}
}
func (*nonStreamingModel) Constraints() llm.ModelConstraints { return llm.ModelConstraints{} }

func (m *nonStreamingModel) Generate(context.Context, *llm.Request) (*llm.Response, error) {
	m.generateCalls++

	return &llm.Response{
		Message:      llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("done")),
		FinishReason: llm.FinishReasonStop,
	}, nil
}

func (m *nonStreamingModel) GenerateEvents(context.Context, *llm.Request) iter.Seq2[llm.Event, error] {
	m.generateEventsCalls++

	return func(yield func(llm.Event, error) bool) {
		yield(nil, errors.New("streaming is unsupported"))
	}
}

func TestGenerateUsesBatchAPIWhenModelDoesNotSupportStreaming(t *testing.T) {
	t.Parallel()

	model := &nonStreamingModel{}
	agentUnderTest := &LLMAgent{}

	response, err := agentUnderTest.generate(
		t.Context(),
		model,
		&llm.Request{},
		func() agent.EventEnvelope { return agent.EventEnvelope{} },
		func(agent.Event, error) bool { return true },
	)

	require.NoError(t, err)
	require.Equal(t, "done", response.Message.TextContent())
	require.Equal(t, 1, model.generateCalls)
	require.Zero(t, model.generateEventsCalls)
}
