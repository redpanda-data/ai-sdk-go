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
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/providers/fireworks"
)

func TestFireworksContractSyntheticCancellationMechanics(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			return
		}

		<-r.Context().Done()
	}))
	defer server.Close()

	c := contractCase{Model: fireworks.Catalog().All()[0].ID, API: catalog.WireProtocolOpenAIChatCompletions, Transport: catalog.WireTransportDirect, Feature: contractCancel}
	e := contractHTTP(t, server.Client(), server.URL, "synthetic-only-not-a-credential", c, contractRequest(t, c), nil)
	require.True(t, e.RequestWritten)
	require.True(t, e.Cancelled)
	require.Empty(t, e.ErrorClass)
	require.Empty(t, e.Response)
}

func TestFireworksContractSyntheticLifecycleBoundaries(t *testing.T) {
	t.Parallel()

	for _, o := range fireworks.Catalog().All() {
		if o.Life.Retires.IsZero() {
			continue
		}

		t.Run(o.ID, func(t *testing.T) {
			t.Parallel()
			before := contractInventoryAt(t, o.Life.Retires.Add(-24*time.Hour))
			on := contractInventoryAt(t, o.Life.Retires)
			found := false

			for _, c := range before {
				if c.Model == o.ID {
					found = true
				}
			}

			require.True(t, found)

			for _, c := range on {
				require.NotEqual(t, o.ID, c.Model, "retired offering must not produce paid live cases")
			}
		})
	}
}

func TestFireworksContractSyntheticRequestControls(t *testing.T) {
	t.Parallel()

	manifest := contractLoadManifest(t)
	for _, evidence := range manifest.Evidence {
		c := evidence.Case
		request := contractRequest(t, c)
		if c.API == catalog.WireProtocolOpenAIResponses {
			require.Equal(t, false, request["store"])
		}

		if c.API == catalog.WireProtocolAnthropicMessages && c.Effort == contractEffortNone {
			require.Equal(t, "disabled", contractObject(request["thinking"])["type"])
			require.Nil(t, contractObject(request["output_config"])["effort"])
		}

		if c.API == catalog.WireProtocolAnthropicMessages && c.Feature == contractStructured {
			require.Equal(t, "json_schema", contractObject(contractObject(request["output_config"])["format"])["type"])
		}

		if c.Feature == contractCancel {
			output := request["max_tokens"]
			if c.API == catalog.WireProtocolOpenAIResponses {
				output = request["max_output_tokens"]
			}

			require.Equal(t, 16, output)
		}
	}
}
