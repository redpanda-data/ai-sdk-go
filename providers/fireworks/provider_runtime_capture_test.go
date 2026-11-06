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
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestFireworksSDKCheckpointKeepsRawToolLinkage(t *testing.T) {
	t.Parallel()

	request := json.RawMessage(`{"messages":[{"role":"user","content":"use lookup"}]}`)
	history := json.RawMessage(`{"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"native-tool-id","name":"lookup","input":{"key":"x"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"native-tool-id","content":"violet-lantern-zebra"}]}]}`)
	first := &sdkRuntimeBody{}
	_, err := first.data.WriteString(`{"content":[{"type":"tool_use","id":"native-tool-id","name":"lookup","input":{"key":"x"}}],"id":"native-message-id"}`)
	require.NoError(t, err)
	wire := &sdkRuntimeWire{
		key: "sdk-checkpoint-not-a-credential", path: filepath.Join(t.TempDir(), "capture.json"),
		report: contractManifest{RunKind: "checkpoint_regression", Evidence: []contractEvidence{
			{Exchanges: []contractExchange{{Request: request, ResponseHeaders: map[string]string{"Content-Type": "application/json", "X-Test-Receipt": "raw-header"}}}},
			{Exchanges: []contractExchange{{Request: history, Response: `{}`, ResponseHeaders: map[string]string{"Content-Type": "application/json"}}}},
		}},
		bodies: []*sdkRuntimeBody{first, nil},
	}
	wire.checkpoint(t)
	original, err := os.ReadFile(wire.path)
	require.NoError(t, err)

	for range 4 {
		wire.checkpoint(t)
		current, err := os.ReadFile(wire.path)
		require.NoError(t, err)
		require.Equal(t, original, current, "repeated checkpoints must not rehash existing IDs or request hashes")
	}

	require.Equal(t, request, wire.report.Evidence[0].Exchanges[0].Request)
	require.Equal(t, history, wire.report.Evidence[1].Exchanges[0].Request)
	require.Equal(t, "raw-header", wire.report.Evidence[0].Exchanges[0].ResponseHeaders["X-Test-Receipt"])

	var report contractManifest
	require.NoError(t, json.Unmarshal(original, &report))
	var response, replay map[string]any
	require.NoError(t, json.Unmarshal([]byte(report.Evidence[0].Exchanges[0].Response), &response))
	require.NoError(t, json.Unmarshal(report.Evidence[1].Exchanges[0].Request, &replay))

	id := contractString(contractObject(contractArray(response["content"])[0])["id"])
	require.NotEqual(t, "native-tool-id", id)

	messages := contractArray(replay["messages"])
	assistant := contractObject(contractArray(contractObject(messages[0])["content"])[0])
	result := contractObject(contractArray(contractObject(messages[1])["content"])[0])

	require.Equal(t, id, assistant["id"])
	require.Equal(t, id, result["tool_use_id"])

	for _, row := range report.Evidence {
		require.Equal(t, []string{contractRequestHash(row.Exchanges[0].Request)}, row.Capture.RequestSHA256)
		require.NotContains(t, row.Exchanges[0].ResponseHeaders, "X-Test-Receipt")
	}
}

func TestFireworksSDKCapturePathsNeverReplaceFixtures(t *testing.T) {
	t.Parallel()
	parent := t.TempDir()

	for _, label := range []string{"messages-live", "messages-live-outcome", "minimax-schema-default", "minimax-schema-adaptive"} {
		path := filepath.Join(parent, label+".json")
		require.NoError(t, os.WriteFile(path, []byte("immutable fixture"), 0o600))
	}

	first := sdkRuntimeCapturePath(t, parent, "messages-live")
	second := sdkRuntimeCapturePath(t, parent, "messages-live")
	require.NotEqual(t, first, second)
	require.Equal(t, "messages-live-incomplete.json", filepath.Base(first))
	require.Equal(t, "messages-live-incomplete.json", filepath.Base(second))
	require.NotEqual(t, parent, filepath.Dir(first))
	require.NotEqual(t, filepath.Dir(first), filepath.Dir(second))

	for _, path := range []string{first, second} {
		wire := &sdkRuntimeWire{key: "sdk-checkpoint-not-a-credential", path: path}
		wire.checkpoint(t)

		_, err := os.Stat(path)
		require.NoError(t, err)
	}

	for _, label := range []string{"messages-live", "messages-live-outcome", "minimax-schema-default", "minimax-schema-adaptive"} {
		data, err := os.ReadFile(filepath.Join(parent, label+".json"))
		require.NoError(t, err)
		require.Equal(t, "immutable fixture", string(data))
	}
}
