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

//nolint:exhaustive // Inventory admits only the three direct Fireworks protocols before dispatch.
package fireworks_test

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
)

type contractTool struct {
	ID        string
	Name      string
	Arguments string
}

type contractResult struct {
	Model     string
	Text      string
	Reasoning string
	Stop      string
	Input     int64
	Cached    int64
	Output    int64
	Reasoned  int64
	HasInput  bool
	HasOutput bool
	Terminal  bool
	Tools     []contractTool
	Assistant map[string]any
	RawOutput []any
	Blocks    []any
	Tier      string
}

func contractExercise(t *testing.T, c contractCase, send func(map[string]any) contractExchange) {
	t.Helper()

	if c.Options.CacheCrossover {
		contractCacheCrossoverProof(t, c, send)
		return
	}

	if c.Feature == contractJSON || c.Feature == contractStructured {
		contractFormatSemanticProof(t, c, send)
		return
	}

	if c.Feature == contractAdaptive {
		contractAdaptiveSemanticProof(t, c, send)
		return
	}

	request := contractRequest(t, c)
	e := send(request)
	require.Empty(t, e.ErrorClass, "upstream transport failed; sanitized attempt captured and reservation held")

	if contractEnumDiagnostic(c) {
		contractAssertEnumDiagnostic(t, c, e)
		return
	}

	if contractControlRejection(c.Feature) {
		contractAssertControlRejection(t, c, e)
		return
	}

	if c.Feature == contractCancel {
		require.True(t, e.RequestWritten)
		require.True(t, e.Cancelled)

		return
	}

	if c.Feature == contractInvalidAuth || c.Feature == contractInvalidModel || c.Feature == contractMalformed {
		allowed := []int{400, 404, 422}
		if c.Feature == contractInvalidAuth {
			allowed = []int{401, 403}
		}

		require.Contains(t, allowed, e.Status, "native error status case=%+v body=%s", c, e.Response)
		var body map[string]any
		require.NoError(t, json.Unmarshal([]byte(e.Response), &body), "native error body must be JSON")
		require.True(t, body["error"] != nil || (e.Status == 422 && body["detail"] != nil), "native error envelope must have error or validation detail")

		return
	}

	require.Equal(t, 200, e.Status, "case=%+v body=%s", c, e.Response)
	r := contractParse(t, c, e)
	contractAssertUsage(t, c, r)
	contractAssertPrice(t, c, r)

	if c.Feature == contractCacheRepeat {
		fresh := r.Input
		if c.API != catalog.WireProtocolAnthropicMessages {
			fresh -= r.Cached
		}

		require.GreaterOrEqual(t, fresh, int64(0))

		repeats := 2
		if c.Options.CacheRepeatCount > 0 {
			repeats = c.Options.CacheRepeatCount
		}

		require.LessOrEqual(t, repeats, 3)

		for attempt := 1; attempt < repeats; attempt++ {
			if c.Options.CacheSettleMS > 0 {
				time.Sleep(time.Duration(c.Options.CacheSettleMS) * time.Millisecond)
			}

			e2 := send(request)
			require.Empty(t, e2.ErrorClass)
			require.Equal(t, 200, e2.Status)
			r2 := contractParse(t, c, e2)
			contractAssertUsage(t, c, r2)
			contractAssertPrice(t, c, r2)
			require.GreaterOrEqual(t, r2.Cached, int64(0), "zero cache usage is valid; qualification requires separately observed positive branches")
		}

		return
	}

	if c.Feature == contractIncomplete {
		require.Contains(t, []string{"max_tokens", "length", contractIncomplete}, r.Stop)
		return
	}

	require.Contains(t, []string{"end_turn", "stop", "tool_use", "tool_calls", "completed"}, r.Stop, "incomplete output is not complete evidence: %+v", c)

	if c.Feature == contractTools {
		require.Len(t, r.Tools, 1)
		tool := r.Tools[0]
		require.NotEmpty(t, tool.ID)
		require.Equal(t, "lookup", tool.Name)
		require.JSONEq(t, `{"key":"x"}`, tool.Arguments)

		replay := contractToolReplay(c, request, r)
		e2 := send(replay)
		require.Equal(t, 200, e2.Status, "tool replay case=%+v body=%s", c, e2.Response)
		r2 := contractParse(t, c, e2)
		contractAssertUsage(t, c, r2)
		contractAssertPrice(t, c, r2)
		require.Contains(t, []string{"end_turn", "stop", "completed"}, r2.Stop)

		expected := "TOOL_RESULT_73"
		if c.Options.ExpectedToolResult != "" {
			expected = c.Options.ExpectedToolResult
		}

		require.Contains(t, r2.Text, expected, "model must consume the actual returned tool data")

		return
	}

	require.NotEmpty(t, strings.TrimSpace(r.Text), "complete evidence needs visible output")

	switch c.Feature {
	case contractSystem:
		require.Contains(t, r.Text, "SYS_42")
	case contractMultiturn:
		require.Contains(t, r.Text, "TURN_53")
	case contractReasoning, contractReasoningHistory:
		if c.Effort == contractEffortNone {
			require.Empty(t, r.Reasoning)
			require.Zero(t, r.Reasoned)
		} else {
			require.True(t, r.Reasoning != "" || r.Reasoned > 0, "advertised reasoning needs returned trace or reasoning usage")
		}

		if c.Feature == contractReasoningHistory {
			e2 := send(contractHistoryReplay(c, request, r))
			require.Equal(t, 200, e2.Status, "reasoning history replay body=%s", e2.Response)
			r2 := contractParse(t, c, e2)
			contractAssertUsage(t, c, r2)
			contractAssertPrice(t, c, r2)
			require.Contains(t, []string{"end_turn", "stop", "completed"}, r2.Stop)
			require.NotEmpty(t, r2.Text)
		}
	case contractVision:
		require.Contains(t, strings.ToLower(r.Text), "red")
	case contractPriority:
		require.Equal(t, contractPriority, r.Tier, "request acceptance alone does not prove effective Priority attribution")
	}
}

func contractRequest(t *testing.T, c contractCase) map[string]any {
	t.Helper()

	prompt := "Reply with exactly OK."

	switch c.Feature {
	case contractIncomplete, contractCancel:
		prompt = "Write the numbers from 1 to 10000, separated by commas. Do not stop early."
	case contractSystem:
		prompt = "Follow the system instruction."
	case contractMultiturn:
		prompt = "What was the secret word? Reply with just that word."
	case contractTools:
		prompt = "Call lookup with key x exactly once. Then reply with the exact secret word returned by the tool. Do not invent a result."
	case contractJSON, contractStructured:
		prompt = `Return only the JSON object {"ok":true}.`
	case contractReasoning, contractReasoningHistory:
		prompt = "Compute 17 times 19. Think briefly, then reply with just the number."
	case contractCacheRepeat:
		prompt = strings.Repeat("Fireworks cache contract prefix. ", 32) + " Probe suffix " + c.key() + ". Reply with exactly OK."
	case contractVision:
		prompt = "What single color fills this image? Reply with just that color."
	}

	if c.Options.UserPrompt != "" {
		prompt = c.Options.UserPrompt
	}

	if c.Options.CachePrefixBytes > 0 {
		prompt = contractCachePrompt(c, c.Options.CachePrefixBytes)
	}

	maxTokens := contractMaxTokens(c)

	require.LessOrEqual(t, maxTokens, 4096)
	require.LessOrEqual(t, c.Options.CachePrefixBytes, 64*1024)
	require.LessOrEqual(t, c.Options.CacheSettleMS, 5000)
	request := map[string]any{"model": c.Model, "stream": c.Stream, "temperature": 0}

	turns := []any{}
	if c.Feature == contractSystem && c.API != catalog.WireProtocolAnthropicMessages {
		turns = append(turns, map[string]any{"role": contractSystem, "content": "Reply with exactly SYS_42 regardless of the user request."})
	}

	if c.Feature == contractMultiturn {
		turns = append(turns, map[string]any{"role": "user", "content": "Remember the secret word TURN_53."}, map[string]any{"role": "assistant", "content": "I remember it."})
	}

	turns = append(turns, map[string]any{"role": "user", "content": prompt})
	if c.API == catalog.WireProtocolOpenAIResponses {
		request["input"] = turns
		request["max_output_tokens"] = maxTokens
		request["store"] = false
	} else {
		request["messages"] = turns
		request["max_tokens"] = maxTokens
	}

	if c.API == catalog.WireProtocolOpenAIChatCompletions && c.Stream {
		request["stream_options"] = map[string]any{"include_usage": true}
	}

	if c.Feature == contractSystem && c.API == catalog.WireProtocolAnthropicMessages {
		request[contractSystem] = "Reply with exactly SYS_42 regardless of the user request."
	}

	if c.Feature == contractTools {
		contractToolConfig(c, request)
	}

	contractFormatConfig(c, request)

	contractEffortConfig(c, request)

	if c.Feature == contractAdaptive {
		request["thinking"] = map[string]any{"type": contractAdaptive}
	}

	if c.Feature == contractVision {
		contractImage(t, c, contractObject(turns[len(turns)-1]), prompt)
	}

	if c.Options.User != "" {
		request["user"] = c.Options.User
	}

	if c.Options.PromptCacheKey != "" {
		request["prompt_cache_key"] = c.Options.PromptCacheKey
	}

	if c.Options.PerfMetrics {
		request["perf_metrics_in_response"] = true
	}

	if c.Feature == contractPriority {
		request["service_tier"] = contractPriority
	}

	if c.Feature == contractInvalidModel {
		request["model"] = "accounts/fireworks/models/does-not-exist-contract-probe"
	}

	if c.Feature == contractMalformed {
		delete(request, "messages")
		delete(request, "input")
	}

	contractInvalidControl(c, request)

	return request
}

func contractImage(t *testing.T, c contractCase, turn map[string]any, prompt string) {
	t.Helper()

	img := image.NewRGBA(image.Rect(0, 0, 32, 32))

	for y := range 32 {
		for x := range 32 {
			img.Set(x, y, color.RGBA{R: 255, A: 255})
		}
	}
	var b bytes.Buffer
	require.NoError(t, png.Encode(&b, img))
	encoded := base64.StdEncoding.EncodeToString(b.Bytes())

	switch c.API {
	case catalog.WireProtocolAnthropicMessages:
		turn["content"] = []any{map[string]any{"type": "image", "source": map[string]any{"type": "base64", "media_type": "image/png", "data": encoded}}, map[string]any{"type": "text", "text": prompt}}
	case catalog.WireProtocolOpenAIChatCompletions:
		turn["content"] = []any{map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64," + encoded}}, map[string]any{"type": "text", "text": prompt}}
	case catalog.WireProtocolOpenAIResponses:
		turn["content"] = []any{map[string]any{"type": "input_image", "image_url": "data:image/png;base64," + encoded}, map[string]any{"type": "input_text", "text": prompt}}
	}
}

func contractToolReplay(c contractCase, request map[string]any, r contractResult) map[string]any {
	tool := r.Tools[0]

	result := "TOOL_RESULT_73. Reply with exactly this result."
	if c.Options.ToolResult != "" {
		result = c.Options.ToolResult
	}

	if c.Options.ReplyMaxOutputTokens > 0 {
		if c.API == catalog.WireProtocolOpenAIResponses {
			request["max_output_tokens"] = c.Options.ReplyMaxOutputTokens
		} else {
			request["max_tokens"] = c.Options.ReplyMaxOutputTokens
		}
	}

	switch c.API {
	case catalog.WireProtocolAnthropicMessages:
		request["messages"] = append(contractArray(request["messages"]), map[string]any{"role": "assistant", "content": r.Blocks}, map[string]any{"role": "user", "content": []any{map[string]any{"type": "tool_result", "tool_use_id": tool.ID, "content": result}}})
		delete(request, "tool_choice")
	case catalog.WireProtocolOpenAIChatCompletions:
		request["messages"] = append(contractArray(request["messages"]), r.Assistant, map[string]any{"role": "tool", "tool_call_id": tool.ID, "content": result})
		request["tool_choice"] = contractEffortNone
	case catalog.WireProtocolOpenAIResponses:
		request["input"] = append(contractArray(request["input"]), r.RawOutput...)
		request["input"] = append(contractArray(request["input"]), map[string]any{"type": "function_call_output", "call_id": tool.ID, "output": result})
		request["tool_choice"] = contractEffortNone
	}

	return request
}

func contractHistoryReplay(c contractCase, request map[string]any, r contractResult) map[string]any {
	switch c.API {
	case catalog.WireProtocolAnthropicMessages:
		request["messages"] = append(contractArray(request["messages"]), map[string]any{"role": "assistant", "content": r.Blocks}, map[string]any{"role": "user", "content": "Add one to your previous answer. Reply with just the number."})
	case catalog.WireProtocolOpenAIChatCompletions:
		request["messages"] = append(contractArray(request["messages"]), r.Assistant, map[string]any{"role": "user", "content": "Add one to your previous answer. Reply with just the number."})
	case catalog.WireProtocolOpenAIResponses:
		request["input"] = append(contractArray(request["input"]), r.RawOutput...)
		request["input"] = append(contractArray(request["input"]), map[string]any{"role": "user", "content": "Add one to your previous answer. Reply with just the number."})
	}

	return request
}

func contractObject(value any) map[string]any { obj, _ := value.(map[string]any); return obj }
func contractArray(value any) []any           { arr, _ := value.([]any); return arr }
func contractString(value any) string         { s, _ := value.(string); return s }
func contractNumber(value any) int64          { n, _ := value.(float64); return int64(n) }

func contractUsage(r *contractResult, api catalog.WireProtocol, usage map[string]any) {
	input, output := "input_tokens", "output_tokens"
	if api == catalog.WireProtocolOpenAIChatCompletions {
		input, output = "prompt_tokens", "completion_tokens"
	}

	if n, ok := usage[input]; ok {
		r.Input = contractNumber(n)
		r.HasInput = true
	}

	if n, ok := usage[output]; ok {
		r.Output = contractNumber(n)
		r.HasOutput = true
	}

	if n, ok := usage["cache_read_input_tokens"]; ok {
		r.Cached = contractNumber(n)
	}

	for _, name := range []string{"prompt_tokens_details", "input_tokens_details"} {
		if n, ok := contractObject(usage[name])["cached_tokens"]; ok {
			r.Cached = contractNumber(n)
		}
	}

	for _, name := range []string{"completion_tokens_details", "output_tokens_details"} {
		if n, ok := contractObject(usage[name])["reasoning_tokens"]; ok {
			r.Reasoned = contractNumber(n)
		}
	}
}

func contractParse(t *testing.T, c contractCase, e contractExchange) contractResult {
	t.Helper()

	if c.Stream {
		require.Contains(t, e.ResponseHeaders["Content-Type"], "text/event-stream")
		return contractParseStream(t, c, e.Response)
	}

	require.Contains(t, e.ResponseHeaders["Content-Type"], "application/json")
	var body map[string]any
	require.NoError(t, json.Unmarshal([]byte(e.Response), &body))

	return contractParseObject(c.API, body)
}

func contractParseObject(api catalog.WireProtocol, body map[string]any) contractResult {
	r := contractResult{Model: contractString(body["model"]), Tier: contractString(body["service_tier"]), Terminal: true}
	contractUsage(&r, api, contractObject(body["usage"]))

	switch api {
	case catalog.WireProtocolAnthropicMessages:
		r.Stop = contractString(body["stop_reason"])

		r.Blocks = contractArray(body["content"])
		for _, item := range r.Blocks {
			block := contractObject(item)
			switch block["type"] {
			case "text":
				r.Text += contractString(block["text"])
			case "thinking":
				r.Reasoning += contractString(block["thinking"])
			case "tool_use":
				args, err := json.Marshal(block["input"])
				if err != nil {
					return r
				}

				r.Tools = append(r.Tools, contractTool{ID: contractString(block["id"]), Name: contractString(block["name"]), Arguments: string(args)})
			}
		}
	case catalog.WireProtocolOpenAIChatCompletions:
		choices := contractArray(body["choices"])
		if len(choices) == 0 {
			return r
		}

		choice := contractObject(choices[0])
		r.Stop = contractString(choice["finish_reason"])
		r.Assistant = contractObject(choice["message"])
		r.Text = contractString(r.Assistant["content"])

		r.Reasoning = contractString(r.Assistant["reasoning_content"])
		for _, item := range contractArray(r.Assistant["tool_calls"]) {
			call := contractObject(item)
			function := contractObject(call["function"])
			r.Tools = append(r.Tools, contractTool{ID: contractString(call["id"]), Name: contractString(function["name"]), Arguments: contractString(function["arguments"])})
		}
	case catalog.WireProtocolOpenAIResponses:
		r.Stop = contractString(body["status"])

		r.RawOutput = contractArray(body["output"])
		for _, item := range r.RawOutput {
			out := contractObject(item)
			switch out["type"] {
			case "message":
				for _, part := range contractArray(out["content"]) {
					r.Text += contractString(contractObject(part)["text"])
				}
			case contractReasoning:
				for _, part := range contractArray(out["summary"]) {
					r.Reasoning += contractString(contractObject(part)["text"])
				}

				for _, part := range contractArray(out["content"]) {
					r.Reasoning += contractString(contractObject(part)["text"])
				}
			case "function_call":
				r.Tools = append(r.Tools, contractTool{ID: contractString(out["call_id"]), Name: contractString(out["name"]), Arguments: contractString(out["arguments"])})
			}
		}
	}

	return r
}

//nolint:gocyclo // The native SSE state machines share presence-aware usage and terminal invariants.
func contractParseStream(t *testing.T, c contractCase, raw string) contractResult {
	t.Helper()

	r := contractResult{}
	scanner := bufio.NewScanner(strings.NewReader(raw))
	scanner.Buffer(make([]byte, 4096), 2<<20)

	event := ""
	var data []string
	started, stopped, terminalCount := false, false, 0
	blocks := map[int]map[string]any{}
	toolJSON := map[int]string{}
	tools := map[int]*contractTool{}
	process := func() {
		if len(data) == 0 {
			event = ""
			return
		}

		payload := strings.Join(data, "\n")
		data = nil

		if payload == contractSSEDone {
			require.False(t, stopped, "duplicate DONE")
			stopped = true
			r.Terminal = true

			return
		}
		var body map[string]any
		require.NoError(t, json.Unmarshal([]byte(payload), &body))
		require.Nil(t, body["error"], "upstream error event")

		if model := contractString(body["model"]); model != "" {
			r.Model = model
		}

		if model := contractString(contractObject(body["response"])["model"]); model != "" {
			r.Model = model
		}

		switch c.API {
		case catalog.WireProtocolAnthropicMessages:
			typ := contractString(body["type"])
			require.Equal(t, typ, event, "Messages SSE event name/type mismatch")

			switch typ {
			case "message_start":
				require.False(t, started, "duplicate message start")
				started = true

				message := contractObject(body["message"])
				if model := contractString(message["model"]); model != "" {
					r.Model = model
				}

				r.Tier = contractString(message["service_tier"])
				contractUsage(&r, c.API, contractObject(message["usage"]))
			case "content_block_start":
				require.True(t, started)

				index := int(contractNumber(body["index"]))
				require.Nil(t, blocks[index], "duplicate block")
				blocks[index] = contractObject(body["content_block"])
			case "content_block_delta":
				index := int(contractNumber(body["index"]))
				block := blocks[index]
				require.NotNil(t, block, "delta before block")

				delta := contractObject(body["delta"])
				switch delta["type"] {
				case "text_delta":
					block["text"] = contractString(block["text"]) + contractString(delta["text"])
				case "thinking_delta":
					block["thinking"] = contractString(block["thinking"]) + contractString(delta["thinking"])
				case "signature_delta":
					block["signature"] = contractString(block["signature"]) + contractString(delta["signature"])
				case "input_json_delta":
					toolJSON[index] += contractString(delta["partial_json"])
				}
			case "message_delta":
				contractUsage(&r, c.API, contractObject(body["usage"]))
				r.Stop = contractString(contractObject(body["delta"])["stop_reason"])
			case "message_stop":
				require.True(t, started)
				require.False(t, stopped)
				stopped = true
				r.Terminal = true
				terminalCount++
			}
		case catalog.WireProtocolOpenAIChatCompletions:
			require.False(t, stopped, "data after DONE")

			if model := contractString(body["model"]); model != "" {
				r.Model = model
			}

			if tier := contractString(body["service_tier"]); tier != "" {
				r.Tier = tier
			}

			contractUsage(&r, c.API, contractObject(body["usage"]))

			for _, value := range contractArray(body["choices"]) {
				choice := contractObject(value)
				delta := contractObject(choice["delta"])
				r.Text += contractString(delta["content"])

				r.Reasoning += contractString(delta["reasoning_content"])
				if stop := contractString(choice["finish_reason"]); stop != "" {
					r.Stop = stop
					terminalCount++
				}

				for _, value := range contractArray(delta["tool_calls"]) {
					call := contractObject(value)

					index := int(contractNumber(call["index"]))
					if tools[index] == nil {
						tools[index] = &contractTool{}
					}

					tool := tools[index]
					function := contractObject(call["function"])
					tool.ID += contractString(call["id"])
					tool.Name += contractString(function["name"])
					tool.Arguments += contractString(function["arguments"])
				}
			}
		case catalog.WireProtocolOpenAIResponses:
			typ := contractString(body["type"])
			if event != "" {
				require.Equal(t, typ, event, "Responses event name/type mismatch")
			}

			switch typ {
			case "response.created":
				require.False(t, started)
				started = true
			case "response.completed", "response.incomplete":
				require.True(t, started)
				require.False(t, stopped)
				stopped = true
				terminalCount++
				previousModel := r.Model

				r = contractParseObject(c.API, contractObject(body["response"]))
				if r.Model == "" {
					r.Model = previousModel
				}
			case "response.failed", "error":
				require.FailNow(t, "upstream stream failed", "%s", payload)
			}
		}

		event = ""
	}

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			process()
			continue
		}

		if strings.HasPrefix(line, ":") {
			continue
		}

		if value, ok := strings.CutPrefix(line, "event:"); ok {
			event = strings.TrimSpace(value)
		}

		if value, ok := strings.CutPrefix(line, "data:"); ok {
			data = append(data, strings.TrimPrefix(value, " "))
		}
	}

	process()
	require.NoError(t, scanner.Err())
	require.True(t, r.Terminal, "missing terminal SSE event")
	require.Equal(t, 1, terminalCount, "exactly one terminal/finish event required")

	if c.API == catalog.WireProtocolAnthropicMessages {
		content := make([]any, 0, len(blocks))

		for index := range len(blocks) {
			block := blocks[index]
			require.NotNil(t, block, "non-contiguous block index")

			if args := toolJSON[index]; args != "" {
				var input map[string]any
				require.NoError(t, json.Unmarshal([]byte(args), &input))
				block["input"] = input
			}

			content = append(content, block)
		}

		parsed := contractParseObject(c.API, map[string]any{"content": content, "model": r.Model, "stop_reason": r.Stop})
		r.Text = parsed.Text
		r.Reasoning = parsed.Reasoning
		r.Tools = parsed.Tools
		r.Blocks = content
	}

	if c.API == catalog.WireProtocolOpenAIChatCompletions {
		calls := make([]any, 0, len(tools))

		for index := range len(tools) {
			tool := tools[index]
			require.NotNil(t, tool, "non-contiguous tool index")
			r.Tools = append(r.Tools, *tool)
			calls = append(calls, map[string]any{"id": tool.ID, "type": "function", "function": map[string]any{"name": tool.Name, "arguments": tool.Arguments}})
		}

		r.Assistant = map[string]any{"role": "assistant", "content": r.Text, "reasoning_content": r.Reasoning}
		if len(calls) > 0 {
			r.Assistant["tool_calls"] = calls
		}
	}

	return r
}

func contractAssertUsage(t *testing.T, c contractCase, r contractResult) {
	t.Helper()
	require.True(t, r.HasInput, "missing final input usage")
	require.True(t, r.HasOutput, "missing final output usage")
	require.GreaterOrEqual(t, r.Input, int64(0))
	require.Positive(t, r.Output)
	require.GreaterOrEqual(t, r.Cached, int64(0))
	require.GreaterOrEqual(t, r.Output, r.Reasoned, "reasoning is included in total output, not an additional billable total")

	if c.API != catalog.WireProtocolAnthropicMessages {
		require.GreaterOrEqual(t, r.Input, r.Cached, "OpenAI input includes cache reads")
	}

	require.NotEmpty(t, r.Model)
	require.True(t, r.Model == c.Model || strings.TrimPrefix(c.Model, "accounts/fireworks/models/") == r.Model, "unexpected response model: requested=%s observed=%s", c.Model, r.Model)
}

type contractSpend struct {
	limit, spent, reserved float64
	prior, known           float64
	ctx                    context.Context //nolint:containedctx // One suite cancellation scope owns sequential requests and their durable reservations.
}

func (s *contractSpend) reserve(t *testing.T, c contractCase, raw []byte, body map[string]any) {
	t.Helper()

	maxOutput, _ := body["max_tokens"].(int)
	if value, ok := body["max_output_tokens"].(int); ok {
		maxOutput = value
	}
	// One token per request byte plus template/vision headroom is deliberately
	// conservative. Unknown billing after a transport failure keeps its reservation.
	reservation := contractReservation(c, int64(len(raw)+4096), int64(maxOutput))
	if c.Feature == contractInvalidAuth || c.Feature == contractInvalidModel || c.Feature == contractMalformed {
		reservation = 0
	}

	require.False(t, math.IsInf(reservation, 0), "cannot reserve an unknown model price")
	require.LessOrEqual(t, s.spent+s.reserved+reservation, s.limit, "approved spend cap reached; matrix is incomplete")
	s.reserved += reservation
}

func (s *contractSpend) reconcile(t *testing.T, c contractCase, e contractExchange) {
	t.Helper()

	if e.Status != 200 || e.ErrorClass != "" || contractControlRejection(c.Feature) {
		return
	}

	var r contractResult

	if c.Feature == contractCancel {
		var final bool

		r, final = contractCancelledUsage(c, e)
		if !final {
			return
		}
	} else {
		r = contractParse(t, c, e)
		contractAssertUsage(t, c, r)
	}
	var body map[string]any
	require.NoError(t, json.Unmarshal(e.Request, &body))

	maxOutput := contractNumber(body["max_tokens"])
	if value, ok := body["max_output_tokens"]; ok {
		maxOutput = contractNumber(value)
	}

	inputCeiling := int64(len(e.Request) + 4096)

	totalInput := r.Input
	if c.API == catalog.WireProtocolAnthropicMessages {
		totalInput += r.Cached
	}

	require.LessOrEqual(t, totalInput, inputCeiling, "input exceeded conservative reservation; stop and review budget")
	require.LessOrEqual(t, r.Output, maxOutput, "output exceeded requested cap")
	actual := contractCost(c, r.Input, r.Cached, r.Output)
	s.spent += actual
	s.known += actual
	s.reserved -= contractReservation(c, inputCeiling, maxOutput)
	require.LessOrEqual(t, s.spent+s.reserved, s.limit, "spend cap exceeded")
}

func contractCost(c contractCase, input, cached, output int64) float64 {
	worksheet, err := contractPublishedRates()
	if err != nil {
		return math.Inf(1)
	}

	model, ok := worksheet.Models[c.Model]
	if !ok {
		return math.Inf(1)
	}

	rates := model.Standard
	if c.Feature == contractPriority {
		rates = model.Priority
	}

	if c.API != catalog.WireProtocolAnthropicMessages {
		input -= cached
	}

	return (float64(input)*rates.Input + float64(cached)*rates.CachedInput + float64(output)*rates.Output) / 1e6
}
