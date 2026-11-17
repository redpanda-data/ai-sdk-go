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

package fireworks_test

import (
	"bufio"
	"encoding/json"
	"maps"
	"regexp"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
)

const (
	contractFormatMarker   = "FORMAT_DISABLED_TOKEN_73"
	contractSchemaMarker   = "SCHEMA_ACTIVE_73"
	contractAdaptiveAnswer = "73579"
	contractAdaptiveTask   = "Find the smallest positive integer n satisfying n mod 101 = 51, n mod 103 = 37, and n mod 107 = 70. Derive the congruence reductions and verify all three remainders. Finish with the numeric result using digits."
)

func contractSemanticOptions(c contractCase) contractRequestOptions {
	switch c.Feature {
	case contractJSON, contractInvalidFormat:
		return contractRequestOptions{FormatProof: true, UserPrompt: "Reply with exactly " + contractFormatMarker + ". Do not output JSON, quotes, braces, or extra text."}
	case contractStructured, contractInvalidSchema:
		return contractRequestOptions{FormatProof: true, UserPrompt: `Return exactly this object, retaining every value, type and field: {"gate":"PROMPT_ONLY","count":"wrong","extra":true}. Do not change it.`}
	case contractAdaptive, contractInvalidAdaptive:
		return contractRequestOptions{AdaptiveProof: true, UserPrompt: contractAdaptiveTask, MaxOutputTokens: 2048}
	default:
		return contractRequestOptions{}
	}
}

func contractControlRejection(feature string) bool {
	return feature == contractInvalidFormat || feature == contractInvalidSchema || feature == contractInvalidAdaptive
}

func contractWithoutFormat(c contractCase, request map[string]any) map[string]any {
	control := maps.Clone(request)

	switch c.API {
	case catalog.WireProtocolAnthropicMessages:
		output := maps.Clone(contractObject(control["output_config"]))
		delete(output, "format")

		if len(output) == 0 {
			delete(control, "output_config")
		} else {
			control["output_config"] = output
		}
	case catalog.WireProtocolOpenAIChatCompletions:
		delete(control, "response_format")
	case catalog.WireProtocolOpenAIResponses:
		delete(control, "text")
	case catalog.WireProtocolGeminiGenerateContent, catalog.WireProtocolBedrockConverse:
		return control
	}

	return control
}

func contractSemanticResponse(t *testing.T, c contractCase, e contractExchange) contractResult {
	t.Helper()
	require.Empty(t, e.ErrorClass)
	require.Equal(t, 200, e.Status, "native semantic request failed: %s", e.Response)
	r := contractParse(t, c, e)
	contractAssertUsage(t, c, r)
	contractAssertPrice(t, c, r)
	require.Contains(t, []string{"end_turn", "stop", "completed"}, r.Stop, "semantic proof needs complete output")
	require.NotEmpty(t, r.Text)

	return r
}

func contractFormatSemanticProof(t *testing.T, c contractCase, send func(map[string]any) contractExchange) {
	t.Helper()
	require.True(t, c.Options.FormatProof, "old prompt-following captures do not qualify formatting enforcement")
	request := contractRequest(t, c)

	plain := contractSemanticResponse(t, c, send(contractWithoutFormat(c, request)))
	if c.Feature == contractJSON {
		expected := contractFormatMarker
		if c.Options.ExpectedPlain != "" {
			expected = c.Options.ExpectedPlain
		}

		require.Equal(t, expected, strings.TrimSpace(plain.Text))
	} else {
		require.JSONEq(t, `{"gate":"PROMPT_ONLY","count":"wrong","extra":true}`, plain.Text)
	}

	controlled := contractSemanticResponse(t, c, send(request))
	var value any
	require.NoError(t, json.Unmarshal([]byte(controlled.Text), &value), "format control must override conflicting content")

	if c.Feature == contractStructured {
		obj := contractObject(value)
		require.Len(t, obj, 2)
		require.Equal(t, contractSchemaMarker, obj["gate"])
		require.JSONEq(t, `{"gate":"SCHEMA_ACTIVE_73","count":73}`, controlled.Text)
	}
}

func contractAdaptiveSemanticProof(t *testing.T, c contractCase, send func(map[string]any) contractExchange) {
	t.Helper()
	require.True(t, c.Options.AdaptiveProof, "an OK smoke response does not qualify adaptive thinking")
	request := contractRequest(t, c)
	disabled := maps.Clone(request)
	disabled["thinking"] = map[string]any{"type": "disabled"}
	on := contractSemanticResponse(t, c, send(request))
	require.True(t, on.Reasoning != "" || on.Reasoned > 0, "explicit adaptive complex-task evidence needs observable native reasoning")
	contractAssertAdaptiveAnswer(t, on.Text)
	off := contractSemanticResponse(t, c, send(disabled))
	require.Empty(t, off.Reasoning, "disabled control must not expose native reasoning")
	require.Zero(t, off.Reasoned)
	contractAssertAdaptiveAnswer(t, off.Text)
}

func contractAssertAdaptiveAnswer(t *testing.T, text string) {
	t.Helper()

	normalized := strings.NewReplacer(",", "", "{", "", "}", "").Replace(text)
	integers := regexp.MustCompile(`[0-9]+`).FindAllString(normalized, -1)
	require.NotEmpty(t, integers)
	require.Equal(t, contractAdaptiveAnswer, integers[len(integers)-1], "the final visible answer must be correct")
}

func contractEnumDiagnostic(c contractCase) bool {
	return c.Feature == contractInvalidFormat || c.Feature == contractInvalidAdaptive
}

func contractAssertEnumDiagnostic(t *testing.T, c contractCase, e contractExchange) {
	t.Helper()

	if e.Status != 200 {
		contractAssertControlRejection(t, c, e)
		return
	}

	require.Empty(t, e.ErrorClass)
	r := contractParse(t, c, e)
	contractAssertUsage(t, c, r)
	contractAssertPrice(t, c, r)
	require.Contains(t, []string{"end_turn", "stop", "completed", "max_tokens", "length", "incomplete"}, r.Stop)
}

func contractAssertControlRejection(t *testing.T, c contractCase, e contractExchange) {
	t.Helper()
	require.Empty(t, e.ErrorClass)
	body := contractControlError(t, c, e)
	require.True(t, body["error"] != nil || body["detail"] != nil, "native structured error required")

	lower := strings.ToLower(e.Response)

	words := []string{"format"}
	if c.Feature == contractInvalidSchema {
		words = []string{"schema", "grammar"}
	}

	if c.Feature == contractInvalidAdaptive || c.Feature == contractAdaptive {
		words = []string{"thinking", "reasoning"}
	}

	matched := false
	for _, word := range words {
		matched = matched || strings.Contains(lower, word)
	}

	require.True(t, matched, "rejection must concern the requested control, not unrelated auth/model errors")
}

func contractControlError(t *testing.T, c contractCase, e contractExchange) map[string]any {
	t.Helper()

	if e.Status != 200 || !c.Stream || c.API != catalog.WireProtocolOpenAIResponses || c.Feature != contractInvalidSchema {
		require.Contains(t, []int{400, 422}, e.Status, "invalid control must be rejected, not silently ignored: %s", e.Response)
		var body map[string]any
		require.NoError(t, json.Unmarshal([]byte(e.Response), &body))

		return body
	}

	require.Contains(t, e.ResponseHeaders["Content-Type"], "text/event-stream")
	scanner := bufio.NewScanner(strings.NewReader(e.Response))
	scanner.Buffer(make([]byte, 4096), 2<<20)
	var native map[string]any
	failed := 0

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		var event map[string]any
		require.NoError(t, json.Unmarshal([]byte(strings.TrimSpace(strings.TrimPrefix(line, "data:"))), &event))
		require.Zero(t, failed, "no events may follow the terminal schema rejection")
		require.Contains(t, []string{"response.created", "response.in_progress", "response.failed"}, event["type"], "a validation rejection cannot generate content or complete")
		response := contractObject(event["response"])
		require.Empty(t, response["output"])

		if event["type"] != "response.failed" {
			require.Nil(t, response["error"])
			continue
		}

		failed++

		require.Equal(t, "failed", response["status"])
		native = contractObject(response["error"])
		require.Equal(t, "invalid_request_error", native["type"])
		require.Contains(t, []any{float64(400), "invalid_request_error"}, native["code"])
	}

	require.NoError(t, scanner.Err())
	require.Equal(t, 1, failed, "exactly one terminal native schema rejection is required")

	return map[string]any{"error": native}
}

func contractInvalidControl(c contractCase, request map[string]any) {
	switch c.Feature {
	case contractInvalidAdaptive:
		request["thinking"] = map[string]any{"type": "sdk_invalid_mode"}
	case contractInvalidSchema, contractInvalidFormat:
		var format map[string]any

		switch c.API {
		case catalog.WireProtocolAnthropicMessages:
			format = contractObject(contractObject(request["output_config"])["format"])
		case catalog.WireProtocolOpenAIChatCompletions:
			format = contractObject(request["response_format"])
		case catalog.WireProtocolOpenAIResponses:
			format = contractObject(contractObject(request["text"])["format"])
		case catalog.WireProtocolGeminiGenerateContent, catalog.WireProtocolBedrockConverse:
			return
		}

		if c.Feature == contractInvalidFormat {
			format["type"] = "sdk_invalid_format"
			return
		}

		bad := map[string]any{"type": "definitely_not_a_json_schema_type"}
		if c.API == catalog.WireProtocolOpenAIChatCompletions {
			contractObject(format["json_schema"])["schema"] = bad
		} else {
			format["schema"] = bad
		}
	}
}
