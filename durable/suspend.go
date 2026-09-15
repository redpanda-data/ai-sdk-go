// Copyright 2026 Redpanda Data, Inc.
//
// Licensed as a Redpanda Enterprise feature, governed by the Redpanda
// Community License Agreement included in the file licenses/RCL.md
//
// Use of this software requires a Redpanda Enterprise license.

package durable

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// suspensionKey marks a tool result as a suspension request. The tool
// registry turns tool errors into error results, so suspension travels in the
// result payload where the durable interceptor can see it intact.
const suspensionKey = "$durable_suspend"

// Suspension asks the durable worker to park the run until external input
// arrives or a timer fires. Tools return it as their result via WaitForInput,
// SleepUntil or Sleep. Outside a durable worker the model simply sees the
// marker object as the tool's result.
type Suspension struct {
	Kind   string    `json:"kind"`
	Name   string    `json:"name,omitempty"`
	FireAt time.Time `json:"fire_at,omitzero"`
}

// Result encodes the suspension as a tool result.
func (s *Suspension) Result() json.RawMessage {
	b, _ := json.Marshal(map[string]*Suspension{suspensionKey: s}) //nolint:errchkjson // static shape

	return b
}

// WaitForInput suspends the run until Client.SendInput(runID, name, payload)
// is called. The payload becomes the tool call's result. Return it directly
// from Tool.Execute.
func WaitForInput(name string) (json.RawMessage, error) {
	return (&Suspension{Kind: AwaitInput, Name: name}).Result(), nil
}

// SleepUntil suspends the run until t. The tool call's result is
// {"fired_at": "<RFC3339>"}.
func SleepUntil(t time.Time) (json.RawMessage, error) {
	return (&Suspension{Kind: AwaitTimer, FireAt: t.UTC()}).Result(), nil
}

// Sleep suspends the run for d.
func Sleep(d time.Duration) (json.RawMessage, error) {
	return SleepUntil(time.Now().Add(d))
}

// ParseSuspension reports whether a tool result is a suspension request.
func ParseSuspension(result json.RawMessage) (*Suspension, bool) {
	if len(result) == 0 || result[0] != '{' {
		return nil, false
	}

	var env map[string]json.RawMessage
	if err := json.Unmarshal(result, &env); err != nil {
		return nil, false
	}

	raw, ok := env[suspensionKey]
	if !ok || len(env) != 1 {
		return nil, false
	}

	var s Suspension
	if err := json.Unmarshal(raw, &s); err != nil {
		return nil, false
	}

	if s.Kind != AwaitInput && s.Kind != AwaitTimer {
		return nil, false
	}

	return &s, true
}

// InputTool returns a tool that suspends the run until input named inputName
// arrives. Use it for approvals, questions to a human, or any external event.
// The model sees description; the delivered payload is returned verbatim as
// the tool result.
func InputTool(toolName, description, inputName string) tool.Tool {
	return &inputTool{
		def: llm.ToolDefinition{
			Name:        toolName,
			Description: description,
			Parameters: json.RawMessage(`{"type":"object","properties":{` +
				`"request":{"type":"string","description":"What you are asking for and why."}},` +
				`"required":["request"]}`),
		},
		inputName: inputName,
	}
}

type inputTool struct {
	def       llm.ToolDefinition
	inputName string
}

func (t *inputTool) Definition() llm.ToolDefinition { return t.def }

func (t *inputTool) Execute(_ context.Context, _ json.RawMessage) (json.RawMessage, error) {
	return WaitForInput(t.inputName)
}

// SleepTool returns a tool the model can call to durably wait. Arguments:
// {"seconds": <number>}. The run holds no worker while sleeping.
func SleepTool() tool.Tool {
	return &sleepTool{def: llm.ToolDefinition{
		Name:        "sleep",
		Description: "Pause this task for the given number of seconds without consuming resources, then continue. Use it to wait for slow external processes or to schedule a follow-up.",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"seconds":{"type":"number","minimum":0}},"required":["seconds"]}`),
	}}
}

type sleepTool struct {
	def llm.ToolDefinition
}

func (t *sleepTool) Definition() llm.ToolDefinition { return t.def }

func (*sleepTool) Execute(_ context.Context, args json.RawMessage) (json.RawMessage, error) {
	var in struct {
		Seconds float64 `json:"seconds"`
	}

	if err := json.Unmarshal(args, &in); err != nil {
		return nil, fmt.Errorf("sleep: invalid arguments: %w", err)
	}

	return Sleep(time.Duration(in.Seconds * float64(time.Second)))
}
