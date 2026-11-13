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
	"maps"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/providers/fireworks"
)

func contractCacheCrossoverProof(t *testing.T, c contractCase, send func(map[string]any) contractExchange) {
	t.Helper()
	require.Equal(t, fireworks.ModelDeepSeekV41Flash1017, c.Model)
	require.Equal(t, catalog.WireProtocolAnthropicMessages, c.API)
	require.Equal(t, contractCacheRepeat, c.Feature)
	require.False(t, c.Stream)

	request := contractRequest(t, c)
	for n, stream := range []bool{true, true, false, true} {
		if n > 0 && c.Options.CacheSettleMS > 0 {
			time.Sleep(time.Duration(c.Options.CacheSettleMS) * time.Millisecond)
		}

		nativeCase := c
		nativeCase.Stream = stream
		body := maps.Clone(request)
		body["stream"] = stream
		e := send(body)
		r := contractSemanticResponse(t, nativeCase, e)
		t.Logf("CROSSOVER call=%d stream=%t input=%d cached=%d output=%d", n+1, stream, r.Input, r.Cached, r.Output)

		if n == 1 && r.Cached == 0 {
			t.Log("INCONCLUSIVE: streaming warm control did not hit; no nonstream or further calls authorized")
			return
		}

		if n == 3 {
			require.Positive(t, r.Cached, "final streaming bracket control must still be warm")
		}
	}
}

func contractVerifyCrossoverCapture(t *testing.T, row contractEvidence) {
	t.Helper()

	if !row.Options.CacheCrossover {
		return
	}

	require.Contains(t, []int{2, 4}, len(row.Exchanges))
	var canonical map[string]any

	for n, e := range row.Exchanges {
		var body map[string]any
		require.NoError(t, json.Unmarshal(e.Request, &body))

		expectedStream := n != 2
		require.Equal(t, expectedStream, body["stream"])
		delete(body, "stream")

		if n == 0 {
			canonical = body
		} else {
			require.Equal(t, canonical, body, "stream is the sole request-body difference")
			require.Equal(t, row.Exchanges[0].RequestHeaders, e.RequestHeaders)
			previous := row.Exchanges[n-1]
			finished := previous.StartedAt.Add(time.Duration(previous.ElapsedMS) * time.Millisecond)
			require.GreaterOrEqual(t, e.StartedAt.Sub(finished).Milliseconds(), int64(2000))
		}

		require.False(t, e.StartedAt.IsZero())
		require.Equal(t, row.Options.AffinityOverride, e.RequestHeaders["X-Session-Affinity"])
		require.Equal(t, row.Options.ClaudeSessionID, e.RequestHeaders["X-Claude-Code-Session-Id"])
		require.Equal(t, row.Options.CacheIsolation, e.RequestHeaders["X-Prompt-Cache-Isolation-Key"])
	}
}
