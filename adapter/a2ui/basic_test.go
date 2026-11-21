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
	"iter"
	"math"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/stretchr/testify/require"
)

const fixtureEast = "east"

func TestBasicSnapshotPartOverA2A(t *testing.T) {
	t.Parallel()

	snapshot := BasicSnapshot{SurfaceID: "overview", Components: []BasicComponent{
		NewColumn("root", "row", "list", "card", "tabs", "modal", "divider", "date"),
		NewRow("row", "text", "image", "icon"),
		NewText("text", "Orders"),
		NewImage("image", "https://example.com/orders.png", "Order volume"),
		NewIcon("icon", "info"),
		NewList("list", "video", "audio"),
		NewVideo("video", "https://example.com/report.mp4"),
		NewAudioPlayer("audio", "https://example.com/report.mp3", "Audio summary"),
		NewCard("card", "field"),
		NewTextField("field", "Filter", ""),
		NewTabs("tabs", Tab{Title: "Totals", Child: "check"}, Tab{Title: "Regions", Child: "choices"}),
		NewCheckBox("check", "Show totals", false),
		NewChoicePicker("choices", "Region", []ChoiceOption{{Label: "East", Value: fixtureEast}}, nil, false),
		NewModal("modal", "button", "slider"),
		NewButton("button", "label", "viewDetails"),
		NewText("label", "Show details"),
		NewSlider("slider", "Percent", 0, 100, 35),
		NewDivider("divider"),
		NewDateTimeInput("date", "Report date", "2026-11-21", DateOnly),
	}}
	handler := a2asrv.AgentExecutorFunc(func(_ context.Context, req *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
		return func(yield func(a2a.Event, error) bool) {
			part, err := BasicSnapshotPart(req.Message, snapshot)
			if err != nil {
				yield(nil, err)
				return
			}

			reply := a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("Orders"))

			reply.ContextID = req.ContextID
			if part != nil {
				reply.Parts = append(reply.Parts, part)
			}

			yield(reply, nil)
		}
	})
	server := httptest.NewServer(a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(handler)))
	t.Cleanup(server.Close)
	client, err := a2aclient.NewFromEndpoints(t.Context(), []*a2a.AgentInterface{
		a2a.NewAgentInterface(server.URL, a2a.TransportProtocolJSONRPC),
	})
	require.NoError(t, err)

	request := a2a.NewMessage(a2a.MessageRoleUser, a2a.NewTextPart("Summarize orders"))
	request.Metadata = offer("1.0", basicCatalogFixture)
	response, err := client.SendMessage(t.Context(), &a2a.SendMessageRequest{Message: request, Tenant: "", Config: nil, Metadata: nil})
	require.NoError(t, err)

	reply, ok := response.(*a2a.Message)
	require.True(t, ok, "unexpected result %T", response)
	require.Len(t, reply.Parts, 2)
	require.Equal(t, "Orders", reply.Parts[0].Text())
	require.Equal(t, mimeFixture, reply.Parts[1].Metadata["mimeType"])
	wire, err := json.Marshal(reply.Parts[1].Data())
	require.NoError(t, err)
	require.JSONEq(t, `[{"version":"1.0","createSurface":{"surfaceId":"overview","catalogId":"https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json","components":[
{"id":"root","component":"Column","children":["row","list","card","tabs","modal","divider","date"]},
{"id":"row","component":"Row","children":["text","image","icon"]},
{"id":"text","component":"Text","text":"Orders"},
{"id":"image","component":"Image","url":"https://example.com/orders.png","description":"Order volume"},
{"id":"icon","component":"Icon","name":"info"},
{"id":"list","component":"List","children":["video","audio"]},
{"id":"video","component":"Video","url":"https://example.com/report.mp4"},
{"id":"audio","component":"AudioPlayer","url":"https://example.com/report.mp3","description":"Audio summary"},
{"id":"card","component":"Card","child":"field"},
{"id":"field","component":"TextField","label":"Filter","value":""},
{"id":"tabs","component":"Tabs","tabs":[{"title":"Totals","child":"check"},{"title":"Regions","child":"choices"}]},
{"id":"check","component":"CheckBox","label":"Show totals","value":false},
{"id":"choices","component":"ChoicePicker","label":"Region","options":[{"label":"East","value":"east"}],"value":[],"variant":"mutuallyExclusive"},
{"id":"modal","component":"Modal","trigger":"button","content":"slider"},
{"id":"button","component":"Button","child":"label","action":{"event":{"name":"viewDetails"}}},
{"id":"label","component":"Text","text":"Show details"},
{"id":"slider","component":"Slider","label":"Percent","min":0,"max":100,"value":35},
{"id":"divider","component":"Divider"},
{"id":"date","component":"DateTimeInput","label":"Report date","value":"2026-11-21","enableDate":true,"enableTime":false}
]}}]`, string(wire))

	request = a2a.NewMessage(a2a.MessageRoleUser, a2a.NewTextPart("Repeat"))
	request.ContextID = reply.ContextID
	require.NotEmpty(t, request.ContextID)
	request.Extensions = []string{ExtensionURI}
	response, err = client.SendMessage(t.Context(), &a2a.SendMessageRequest{Message: request, Tenant: "", Config: nil, Metadata: nil})
	require.NoError(t, err)

	reply, ok = response.(*a2a.Message)
	require.True(t, ok, "unexpected result %T", response)
	require.Len(t, reply.Parts, 1)
	require.Equal(t, "Orders", reply.Parts[0].Text())
}

func TestBasicSnapshotPartRejectsInvalidSnapshots(t *testing.T) {
	t.Parallel()

	message := a2a.NewMessage(a2a.MessageRoleUser, a2a.NewTextPart("Report"))
	message.Metadata = offer("1.0", basicCatalogFixture)
	option := []ChoiceOption{{Label: "East", Value: fixtureEast}, {Label: "West", Value: "west"}}
	large := make([]BasicComponent, 0, 7)

	large = append(large, NewColumn("root", "0", "1", "2", "3", "4", "5"))
	for i := range 6 {
		large = append(large, NewText(strconv.Itoa(i), strings.Repeat("<", 2048)))
	}
	var zero BasicComponent

	for _, tc := range []struct {
		name  string
		nodes []BasicComponent
	}{
		{"zero value", []BasicComponent{zero}},
		{"duplicate ID", []BasicComponent{NewText("root", "Orders"), NewDivider("root")}},
		{"orphan", []BasicComponent{NewText("root", "Orders"), NewDivider("other")}},
		{"missing tab child", []BasicComponent{NewTabs("root", Tab{Title: "Missing", Child: "other"})}},
		{"cycle", []BasicComponent{NewCard("root", "root")}},
		{"unknown icon", []BasicComponent{NewIcon("root", "unknown")}},
		{"HTTP media", []BasicComponent{NewVideo("root", "http://example.com/video")}},
		{"media credentials", []BasicComponent{NewImage("root", "https://user@example.com/image", "")}},
		{"missing event", []BasicComponent{NewButton("root", "label", ""), NewText("label", "Details")}},
		{"wrong date mode", []BasicComponent{NewDateTimeInput("root", "Date", "10:00", DateOnly)}},
		{"date-time without timezone", []BasicComponent{NewDateTimeInput("root", "Date", "2026-11-21T10:00:00", DateAndTime)}},
		{"single choice with two values", []BasicComponent{NewChoicePicker("root", "", option, []string{fixtureEast, "west"}, false)}},
		{"duplicate selected value", []BasicComponent{NewChoicePicker("root", "", option, []string{fixtureEast, fixtureEast}, true)}},
		{"missing selected value", []BasicComponent{NewChoicePicker("root", "", option, []string{"unknown"}, true)}},
		{"oversized choice label", []BasicComponent{NewChoicePicker("root", "", []ChoiceOption{{Label: strings.Repeat("a", 257), Value: fixtureEast}}, nil, false)}},
		{"nonfinite slider", []BasicComponent{NewSlider("root", "Percent", 0, 100, math.NaN())}},
		{"out-of-range slider", []BasicComponent{NewSlider("root", "Percent", 0, 100, 101)}},
		{"oversize encoded surface", large},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			snapshot := BasicSnapshot{SurfaceID: "report", Components: tc.nodes}
			part, err := BasicSnapshotPart(message, snapshot)
			require.ErrorIs(t, err, errInvalidSnapshot)
			require.Nil(t, part)
		})
	}
}

func TestBasicComponentOwnsChildLists(t *testing.T) {
	t.Parallel()

	children := []string{"name"}
	snapshot := BasicSnapshot{SurfaceID: "report", Components: []BasicComponent{
		NewColumn("root", children...), NewText("name", "Orders"),
	}}
	children[0] = "modified by caller"
	message := a2a.NewMessage(a2a.MessageRoleUser)
	message.Metadata = offer("1.0", basicCatalogFixture)
	first, err := BasicSnapshotPart(message, snapshot)
	require.NoError(t, err)

	wire, ok := first.Data().([]any)
	require.True(t, ok)
	envelope, ok := wire[0].(map[string]any)
	require.True(t, ok)
	surface, ok := envelope["createSurface"].(map[string]any)
	require.True(t, ok)
	nodes, ok := surface["components"].([]any)
	require.True(t, ok)
	root, ok := nodes[0].(map[string]any)
	require.True(t, ok)
	outChildren, ok := root["children"].([]any)
	require.True(t, ok)
	require.Equal(t, []any{"name"}, outChildren)
	outChildren[0] = "modified by previous output"
	next, err := BasicSnapshotPart(message, snapshot)
	require.NoError(t, err)
	encoded, err := json.Marshal(next.Data())
	require.NoError(t, err)
	require.Contains(t, string(encoded), `["name"]`)
}
