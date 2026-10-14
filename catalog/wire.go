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

package catalog

import "cmp"

// WireProtocol identifies a request/response API contract, independently of
// the provider serving it. Support does not imply feature parity between APIs.
type WireProtocol string

const (
	WireProtocolOpenAIChatCompletions WireProtocol = "openai_chat_completions"
	WireProtocolOpenAIResponses       WireProtocol = "openai_responses"
	WireProtocolAnthropicMessages     WireProtocol = "anthropic_messages"
	WireProtocolGeminiGenerateContent WireProtocol = "gemini_generate_content"
	WireProtocolBedrockConverse       WireProtocol = "bedrock_converse"
)

// WireTransport identifies the provider binding for a protocol, not an SDK
// mapper or a guarantee that the SDK implements that binding.
type WireTransport string

const (
	// WireTransportDirect is the provider's ordinary protocol endpoint,
	// including documented compatibility endpoints.
	WireTransportDirect WireTransport = "direct"
	// WireTransportBedrockRuntime binds the protocol to bedrock-runtime's
	// protocol endpoint, including its direct Anthropic Messages endpoint.
	WireTransportBedrockRuntime WireTransport = "bedrock_runtime"
	// WireTransportBedrockInvokeModel uses bedrock-runtime's InvokeModel
	// envelope. Anthropic Messages requires the Bedrock anthropic_version
	// body field, so a native Messages request cannot be forwarded unchanged.
	WireTransportBedrockInvokeModel WireTransport = "bedrock_invoke_model"
	// WireTransportBedrockMantle binds the protocol to bedrock-mantle.
	WireTransportBedrockMantle WireTransport = "bedrock_mantle"
	// WireTransportVertex uses Vertex's provider-specific endpoint and
	// version envelope; Anthropic Messages is not byte-identical to native.
	WireTransportVertex WireTransport = "vertex"
)

// WireAPI is one protocol and provider transport accepted by an offering.
// It is discovery metadata: it does not select the SDK's runtime transport.
type WireAPI struct {
	Protocol  WireProtocol
	Transport WireTransport
}

func compareWireAPIs(a, b WireAPI) int {
	if d := cmp.Compare(a.Protocol, b.Protocol); d != 0 {
		return d
	}

	return cmp.Compare(a.Transport, b.Transport)
}

func validateWireAPIs(apis []WireAPI, fail func(string, ...any)) {
	seen := make(map[WireAPI]struct{}, len(apis))

	for i, api := range apis {
		switch api.Protocol {
		case WireProtocolOpenAIChatCompletions, WireProtocolOpenAIResponses,
			WireProtocolAnthropicMessages, WireProtocolGeminiGenerateContent,
			WireProtocolBedrockConverse:
		default:
			fail("WireAPIs[%d].Protocol %q is not a valid wire protocol", i, api.Protocol)
		}

		switch api.Transport {
		case WireTransportDirect, WireTransportBedrockRuntime, WireTransportBedrockInvokeModel,
			WireTransportBedrockMantle, WireTransportVertex:
		default:
			fail("WireAPIs[%d].Transport %q is not a valid wire transport", i, api.Transport)
		}

		if _, duplicate := seen[api]; duplicate {
			fail("WireAPIs[%d] duplicates protocol %q and transport %q", i, api.Protocol, api.Transport)
		}

		seen[api] = struct{}{}
	}
}
