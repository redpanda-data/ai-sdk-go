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

package a2a

// InputModes returns the MIME types the executor's A2A conversion supports as
// model input. Text formats must arrive as TextPart content (decode text uploads
// before sending); application/json may also arrive as a DataPart, serialized to
// text for the model. FilePart bytes and URIs are only serialized as JSON, not
// decoded or fetched, so native file/media types are not advertised.
//
// This is the adapter's execution ceiling, not model catalog metadata. Vision or
// audio model flags do not widen it. Hosts may restrict it further for their
// agents and should validate incoming part representations separately.
// The returned slice is caller-owned and suitable for AgentCard.DefaultInputModes.
func (*Executor) InputModes() []string {
	return []string{"text/plain", "text/csv", "text/markdown", "application/json"}
}

// OutputModes returns the MIME types produced by the executor's A2A conversion:
// TextPart responses and DataPart tool events. It does not promise a structured
// final model response, native media output, or tool execution support.
// The returned slice is caller-owned and suitable for AgentCard.DefaultOutputModes.
func (*Executor) OutputModes() []string {
	return []string{"text/plain", "application/json"}
}
