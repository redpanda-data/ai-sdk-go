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

package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/tool/builtin/webfetch"
)

func TestFetch(t *testing.T) {
	t.Parallel()

	var userAgent string

	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		userAgent = r.Header.Get("User-Agent")

		switch r.URL.Path {
		case "/ok.md":
			w.Header().Set("Content-Type", "text/markdown; charset=utf-8")
			_, _ = w.Write([]byte("# Claude"))
		case "/moved":
			http.Redirect(w, r, "/ok.md", http.StatusFound)
		case "/big":
			w.Header().Set("Content-Type", "text/html")
			_, _ = w.Write([]byte(strings.Repeat("x", 2048)))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)

	parsed, err := url.Parse(server.URL)
	require.NoError(t, err)

	port, err := strconv.Atoi(parsed.Port())
	require.NoError(t, err)

	fetch := newFetcher(webfetch.WithDenyPrivateIPs(false), webfetch.WithInsecureSkipVerify(true),
		webfetch.WithAllowedPorts([]int{port}), webfetch.WithMaxBytes(1024))

	body, media, err := fetch(t.Context(), server.URL+"/ok.md")
	require.NoError(t, err)
	assert.Equal(t, "# Claude", string(body))
	assert.Equal(t, "text/markdown", media)
	assert.NotContains(t, userAgent, "Go-http-client", "Google serves a stub to Go's default User-Agent")

	for path, want := range map[string]string{"/missing": "404", "/moved": "redirect", "/big": "cut off"} {
		_, _, err := fetch(t.Context(), server.URL+path)
		assert.ErrorContains(t, err, want, path)
	}
}

func TestFingerprintPage(t *testing.T) {
	t.Parallel()

	body := []byte("# Claude\n" + strings.Repeat("Claude line\n", 12))
	fetch := func(media string) fetcher {
		return func(_ context.Context, _ string) ([]byte, string, error) { return body, media, nil }
	}
	p := page{ID: "anthropic-pricing", Type: typeMarkdown, URL: "https://platform.claude.com/pricing.md"}

	got, text := fingerprintPage(t.Context(), p, fetch("text/markdown"))
	assert.Equal(t, hash(cleanLines(string(body))), got.SHA256)
	assert.Contains(t, text, "Claude line")

	got, _ = fingerprintPage(t.Context(), p, fetch("text/html"))
	assert.Empty(t, got.SHA256)
	assert.Contains(t, got.Error, "text/html", "a markdown page served as HTML is an error")
}

func TestDecide(t *testing.T) {
	t.Parallel()

	now := time.Date(2026, 10, 15, 12, 0, 0, 0, time.UTC)
	baseline := func() fingerprints {
		return fingerprints{
			Version: 1, Provider: "anthropic", FetchedAt: now.Add(-time.Hour),
			Sources: map[string]source{"a": {SHA256: "1"}, "b": {SHA256: "2"}},
			Inputs:  map[string]string{"RULES.md": "3"},
		}
	}

	tests := []struct {
		name        string
		noBaseline  bool
		editBase    func(f *fingerprints)
		editNow     func(f *fingerprints)
		editOpts    func(o *options)
		wantVerdict string
		wantRun     bool
		wantSame    bool
		wantIDs     []string
	}{
		{name: "forced, with nothing changed", editOpts: func(o *options) { o.Force = true }, wantVerdict: verdictForced, wantRun: true, wantSame: true},
		{name: "no baseline", noBaseline: true, wantVerdict: verdictNoBaseline, wantRun: true},
		{name: "another provider's baseline", editBase: func(f *fingerprints) { f.Provider = "openai" }, wantVerdict: verdictNoBaseline, wantRun: true},
		{name: "a page couldn't be read", editNow: func(f *fingerprints) { f.Sources["b"] = source{Error: "status 404"} }, wantVerdict: verdictUnreadable, wantRun: true, wantIDs: []string{"b"}},
		{name: "a page changed", editNow: func(f *fingerprints) { f.Sources["a"] = source{SHA256: "9"} }, wantVerdict: verdictChanged, wantRun: true, wantIDs: []string{"a"}},
		{name: "an input changed", editNow: func(f *fingerprints) { f.Inputs["RULES.md"] = "9" }, wantVerdict: verdictChanged, wantRun: true, wantIDs: []string{"RULES.md"}},
		{name: "a page is new", editNow: func(f *fingerprints) { f.Sources["c"] = source{SHA256: "4"} }, wantVerdict: verdictChanged, wantRun: true, wantIDs: []string{"c"}},
		{name: "the baseline is too old", editBase: func(f *fingerprints) { f.FetchedAt = now.Add(-61 * time.Hour) }, wantVerdict: verdictStale, wantRun: true, wantSame: true},
		{name: "the baseline has no time", editBase: func(f *fingerprints) { f.FetchedAt = time.Time{} }, wantVerdict: verdictStale, wantRun: true, wantSame: true},
		{name: "unchanged in shadow mode runs anyway", wantVerdict: verdictUnchanged, wantRun: true, wantSame: true},
		{name: "unchanged in enforce mode skips", editOpts: func(o *options) { o.Mode = modeEnforce }, wantVerdict: verdictUnchanged, wantRun: false, wantSame: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			current, base := baseline(), baseline()
			opts := options{Mode: modeShadow, MaxAge: 60 * time.Hour, Now: now}

			for edit, target := range map[*func(*fingerprints)]*fingerprints{&tt.editNow: &current, &tt.editBase: &base} {
				if *edit != nil {
					(*edit)(target)
				}
			}

			if tt.editOpts != nil {
				tt.editOpts(&opts)
			}

			var baselinePtr *fingerprints
			if !tt.noBaseline {
				baselinePtr = &base
			}

			got := decide(current, baselinePtr, opts)
			assert.Equal(t, tt.wantVerdict, got.Verdict)
			assert.Equal(t, tt.wantRun, got.RunAgent)
			assert.Equal(t, tt.wantSame, got.Same, "Same records the match whatever the verdict")
			assert.Equal(t, tt.wantIDs, got.IDs)
		})
	}
}

func TestSummary(t *testing.T) {
	t.Parallel()

	now := time.Date(2026, 10, 15, 12, 0, 0, 0, time.UTC)
	current := fingerprints{Version: 1, Provider: "anthropic", FetchedAt: now, Sources: map[string]source{"anthropic-pricing": {SHA256: "1"}}}
	opts := options{Mode: modeShadow, MaxAge: time.Hour, Now: now}

	shadow := summary(decide(current, &current, opts), "https://github.com/o/r/actions/runs/7")
	assert.Contains(t, shadow, "**Would skip:**")
	assert.Contains(t, shadow, "[run 7](https://github.com/o/r/actions/runs/7)")

	hostile := current
	hostile.Sources = map[string]source{"anthropic-pricing": {SHA256: "1"}, "<img src=x>": {SHA256: "2"}}

	text := summary(decide(current, &hostile, opts), "")
	assert.NotContains(t, text, "<img", "an id only the baseline has is counted, never printed")
	assert.Contains(t, text, "1 page the baseline had is no longer listed")
}

func TestHashInputs(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	require.NoError(t, os.MkdirAll(filepath.Join(dir, "skill", "references"), 0o750))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "skill", "SKILL.md"), []byte("a"), 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "skill", "references", "new.md"), []byte("b"), 0o600))

	inputs := hashInputs(dir, []string{"skill", "gone"})
	assert.Len(t, inputs, 3, "every file under a folder, and the missing path")
	assert.Contains(t, inputs, "skill/references/new.md", "a file the skill starts using is covered")
	assert.Contains(t, inputs["gone"], "unreadable")
}
