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

package a2ui

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/stretchr/testify/require"
)

const (
	basicCatalogFixture = "https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json"
	capabilitiesFixture = "a2uiRendererCapabilities"
	mimeFixture         = "application/a2ui+json"
)

func offer(version string, ids ...any) map[string]any {
	return map[string]any{capabilitiesFixture: map[string]any{version: map[string]any{"supportedCatalogIds": ids}}}
}

func textSnapshot() Snapshot {
	return Snapshot{SurfaceID: "overview", Components: []Component{
		{ID: "root", Kind: Column, Children: []string{"summary"}},
		{ID: "summary", Kind: Text, Text: "17 orders"},
	}}
}

func TestSnapshotPartOverA2A(t *testing.T) {
	t.Parallel()

	exec := a2asrv.AgentExecutorFunc(func(_ context.Context, req *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
		return func(yield func(a2a.Event, error) bool) {
			reply := a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("17 orders"))
			reply.ContextID = req.ContextID

			part, err := SnapshotPart(req.Message, Snapshot{
				SurfaceID:  req.Message.ID,
				Components: textSnapshot().Components,
			})
			if err != nil {
				yield(nil, err)
				return
			}

			if part != nil {
				reply.Parts = append(reply.Parts, part)
			}

			yield(reply, nil)
		}
	})
	srv := httptest.NewServer(a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(exec)))
	t.Cleanup(srv.Close)
	client, err := a2aclient.NewFromEndpoints(t.Context(), []*a2a.AgentInterface{
		a2a.NewAgentInterface(srv.URL, a2a.TransportProtocolJSONRPC),
	})
	require.NoError(t, err)

	send := func(request *a2a.Message) *a2a.Message {
		t.Helper()
		result, err := client.SendMessage(t.Context(), &a2a.SendMessageRequest{Message: request})
		require.NoError(t, err)

		reply, ok := result.(*a2a.Message)
		require.True(t, ok, "unexpected result %T", result)
		require.NotEmpty(t, reply.Parts)
		require.Equal(t, "17 orders", reply.Parts[0].Text())

		return reply
	}

	first := a2a.NewMessage(a2a.MessageRoleUser, a2a.NewTextPart("Summarize orders"))
	first.Metadata = offer("1.0", basicCatalogFixture)
	reply := send(first)
	require.Len(t, reply.Parts, 2)
	require.Equal(t, mimeFixture, reply.Parts[1].Metadata["mimeType"])
	b, err := json.Marshal(reply.Parts[1].Data())
	require.NoError(t, err)
	require.JSONEq(t, `[{"version":"1.0","createSurface":{"surfaceId":"`+first.ID+`","catalogId":"`+basicCatalogFixture+`","components":[{"id":"root","component":"Column","children":["summary"]},{"id":"summary","component":"Text","text":"17 orders"}]}}]`, string(b))

	// A later turn in the same context must not inherit the previous offer.
	second := a2a.NewMessage(a2a.MessageRoleUser, a2a.NewTextPart("Repeat"))
	second.ContextID = reply.ContextID
	require.NotEmpty(t, second.ContextID)
	second.Extensions = []string{"https://a2ui.org/a2a-extension/a2ui/v1.0"}
	require.Len(t, send(second).Parts, 1)
}

func TestAgentExtension(t *testing.T) {
	t.Parallel()

	ext := AgentExtension()
	require.Equal(t, "https://a2ui.org/a2a-extension/a2ui/v1.0", ext.URI)
	require.False(t, ext.Required)
	require.Equal(t, map[string]any{"supportedCatalogIds": []any{basicCatalogFixture}}, ext.Params)
}

func TestSnapshotPartNegotiation(t *testing.T) {
	t.Parallel()

	data := a2a.NewDataPart(map[string]any{"context": "example"})

	data.Metadata = offer("1.0", basicCatalogFixture)
	for _, tc := range []struct {
		name string
		meta map[string]any
		part *a2a.Part
		want bool
	}{
		{name: "wrong version", meta: offer("0.9", basicCatalogFixture)},
		{name: "wrong catalog", meta: offer("1.0", "https://example.com/catalog")},
		{name: "invalid catalog list", meta: offer("1.0", basicCatalogFixture, 1)},
		{name: "DataPart fallback", part: data, want: true},
		{name: "message takes precedence", meta: offer("1.0", "other"), part: data},
		{name: "malformed message capability", meta: map[string]any{capabilitiesFixture: true}, part: data},
		{name: "Go typed catalog IDs", meta: map[string]any{capabilitiesFixture: map[string]any{"1.0": map[string]any{"supportedCatalogIds": []string{basicCatalogFixture}}}}, want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			in := a2a.NewMessage(a2a.MessageRoleUser, a2a.NewTextPart("Hello"))

			in.Metadata = tc.meta
			if tc.part != nil {
				in.Parts = append(in.Parts, tc.part)
			}

			part, err := SnapshotPart(in, textSnapshot())
			require.NoError(t, err)
			require.Equal(t, tc.want, part != nil)
		})
	}
}

func TestSnapshotPartRejectsInvalidSnapshots(t *testing.T) {
	t.Parallel()

	in := a2a.NewMessage(a2a.MessageRoleUser, a2a.NewTextPart("Hello"))
	in.Metadata = offer("1.0", basicCatalogFixture)

	for _, tc := range []struct {
		name string
		edit func(*Snapshot)
	}{
		{name: "empty surface ID", edit: func(s *Snapshot) { s.SurfaceID = "" }},
		{name: "missing root", edit: func(s *Snapshot) { s.Components = s.Components[1:] }},
		{name: "unknown kind", edit: func(s *Snapshot) { s.Components[1].Kind = "Button" }},
		{name: "duplicate ID", edit: func(s *Snapshot) { s.Components[1].ID = "root" }},
		{name: "missing child", edit: func(s *Snapshot) { s.Components[0].Children = []string{"absent"} }},
		{name: "repeated child", edit: func(s *Snapshot) { s.Components[0].Children = []string{"summary", "summary"} }},
		{name: "cycle", edit: func(s *Snapshot) { s.Components[0].Children = []string{"root"} }},
		{name: "unreachable component", edit: func(s *Snapshot) {
			s.Components = append(s.Components, Component{ID: "other", Kind: Text, Text: "not rendered"})
		}},
		{name: "Text used as a container", edit: func(s *Snapshot) { s.Components[1].Children = []string{"root"} }},
		{name: "oversized text", edit: func(s *Snapshot) { s.Components[1].Text = strings.Repeat("x", 1<<16) }},
		{name: "oversized encoded payload", edit: func(s *Snapshot) {
			// Each Text is within its limit, but JSON escaping exceeds the envelope limit.
			for i := range 8 {
				id := fmt.Sprintf("row-%d", i)
				s.Components[0].Children = append(s.Components[0].Children, id)
				s.Components = append(s.Components, Component{ID: id, Kind: Text, Text: strings.Repeat("<", 2048)})
			}
		}},
		{name: "excessive depth", edit: func(s *Snapshot) {
			s.Components[0].Children = []string{"row-0"}
			for i := range 9 {
				s.Components = append(s.Components, Component{
					ID: fmt.Sprintf("row-%d", i), Kind: Column, Children: []string{fmt.Sprintf("row-%d", i+1)},
				})
			}

			s.Components[len(s.Components)-1].Children = []string{"summary"}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			snapshot := textSnapshot()
			tc.edit(&snapshot)
			part, err := SnapshotPart(in, snapshot)
			require.Error(t, err)
			require.Nil(t, part)
		})
	}
}
