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

package toolsearch

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func def(name, group, description, params string) llm.ToolDefinition {
	if params == "" {
		params = `{"type":"object"}`
	}

	return llm.ToolDefinition{
		Name: name, Description: description, Parameters: json.RawMessage(params),
		Deferred: true, Group: llm.ToolGroup{Name: group},
	}
}

// corpus is a small service desk: names, descriptions, argument names and enum
// values are all searchable, and plurals fold onto singulars.
func corpus() []llm.ToolDefinition {
	return []llm.ToolDefinition{
		def("confluence__get_page", "confluence", "Fetch a Confluence page by id or title.", ""),
		def("jira__create_issue", "jira", "Create a Jira issue in a project.",
			`{"type":"object","properties":{"project":{"type":"string"},"issue_type":{"type":"string","enum":["Bug","Story","Task"]}}}`),
		def("jira__search_issues", "jira", "Search Jira issues with JQL.", ""),
		def("servicenow__close_incident", "servicenow", "Close a ServiceNow incident with a resolution code.", ""),
		def("servicenow__create_incident", "servicenow", "Open a ServiceNow incident on behalf of a caller.",
			`{"type":"object","properties":{"caller_sys_id":{"type":"string","description":"sys_id of the affected user"},"urgency":{"type":"string","enum":["low","medium","high"]}}}`),
	}
}

func TestParse(t *testing.T) {
	t.Parallel()

	tests := []struct {
		raw  string
		want Query
	}{
		{"select:jira__create_issue,servicenow__close_incident", Query{Selected: []string{"jira__create_issue", "servicenow__close_incident"}}},
		{"  select: a , , b  ", Query{Selected: []string{"a", "b"}}},
		{"SELECT:a", Query{Selected: []string{"a"}}},
		{"select:", Query{Selected: []string{}}},
		{"+servicenow open an incident", Query{Required: "servicenow", Terms: []string{"open", "an", "incident"}}},
		{"+jira +bug", Query{Required: "jira", Terms: []string{"bug"}}},
		{"createIncident, urgency", Query{Terms: []string{"create", "incident", "urgency"}}},
		{"+ incident", Query{Terms: []string{"incident"}}},
		// Repeated terms are deduplicated; document frequency is counted per term.
		{"incidents incident close", Query{Terms: []string{"incident", "close"}}},
	}

	for _, tt := range tests {
		t.Run(tt.raw, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, Parse(tt.raw))
		})
	}
}

func TestTokenize(t *testing.T) {
	t.Parallel()

	tests := map[string][]string{
		"":                                   nil,
		"servicenow__create_incident":        {"servicenow", "create", "incident"},
		"createIncidentNow":                  {"create", "incident", "now"},
		"getHTTPServerID":                    {"get", "httpserver", "id"},
		"oauth2Token":                        {"oauth2", "token"},
		"a.b-c/d":                            {"a", "b", "c", "d"},
		"search_issues incidents priorities": {"search", "issue", "incident", "priority"},
		"status class analysis":              {"status", "class", "analysis"},
		"ids as":                             {"ids", "as"},
	}

	for input, want := range tests {
		t.Run(input, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, want, tokenize(input))
		})
	}
}

func TestRank(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		query     string
		limit     int
		wantFirst string
		wantAll   []string
		wantSet   []string
		wantEmpty bool
	}{
		{name: "name match wins", query: "close incident", wantFirst: "servicenow__close_incident"},
		{name: "description only", query: "jql", wantAll: []string{"jira__search_issues"}},
		{name: "argument name", query: "caller_sys_id", wantFirst: "servicenow__create_incident"},
		{name: "enum value", query: "urgency high", wantFirst: "servicenow__create_incident"},
		{name: "required term filters the name", query: "+jira issue", wantSet: []string{"jira__create_issue", "jira__search_issues"}},
		{name: "bare required term lists names", query: "+confluence", wantAll: []string{"confluence__get_page"}},
		{name: "repeated terms do not annihilate", query: "incident incidents", wantSet: []string{"servicenow__close_incident", "servicenow__create_incident"}},
		{name: "no match", query: "kubernetes helm chart", wantEmpty: true},
		{name: "limit truncates", query: "incident", limit: 1, wantAll: []string{"servicenow__close_incident"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := Rank(corpus(), Parse(tt.query), tt.limit)

			if tt.wantEmpty {
				assert.Empty(t, got)
				return
			}

			require.NotEmpty(t, got)

			if tt.wantFirst != "" {
				assert.Equal(t, tt.wantFirst, got[0])
			}

			if tt.wantAll != nil {
				assert.Equal(t, tt.wantAll, got)
			}

			if tt.wantSet != nil {
				assert.ElementsMatch(t, tt.wantSet, got)
			}
		})
	}
}

// TestRankIsDeterministic: equal scores break ties by name, so an unchanged
// corpus and query always rank the same way.
func TestRankIsDeterministic(t *testing.T) {
	t.Parallel()

	defs := make([]llm.ToolDefinition, 0, 12)
	for i := range 12 {
		defs = append(defs, def(fmt.Sprintf("svc__identical_%02d", i), "svc", "Identical description for every tool.", ""))
	}

	query := Parse("identical description")
	first := Rank(defs, query, 5)
	assert.Equal(t, []string{"svc__identical_00", "svc__identical_01", "svc__identical_02", "svc__identical_03", "svc__identical_04"}, first)

	for range 20 {
		assert.Equal(t, first, Rank(defs, query, 5))
	}
}

// TestRankUsesGroupDescription: a domain word that appears only in the group's
// description still finds the group's tools.
func TestRankUsesGroupDescription(t *testing.T) {
	t.Parallel()

	gcal := def("gcal__create_event", "gcal", "Create an event.", "")
	gcal.Group.Description = "Google Calendar events and invitations"

	assert.Equal(t, []string{"gcal__create_event"},
		Rank([]llm.ToolDefinition{gcal, def("jira__create_issue", "jira", "Create an issue.", "")}, Parse("calendar"), 5))
}
