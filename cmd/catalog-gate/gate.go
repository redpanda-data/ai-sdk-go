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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/redpanda-data/ai-sdk-go/tool/builtin/webfetch"
)

// Gate modes: shadow reports what it would do and always runs the agent.
const (
	modeShadow  = "shadow"
	modeEnforce = "enforce"
)

// Verdicts, in the order decide checks them.
const (
	verdictForced     = "forced"
	verdictNoBaseline = "no-baseline"
	verdictUnreadable = "unreadable"
	verdictChanged    = "changed"
	verdictStale      = "stale"
	verdictUnchanged  = "unchanged"
)

// inputPaths are the run's inputs besides the pages, files or whole folders:
// a change to any file in them changes what the agent would do, so a new file
// the prompt or a skill starts using is covered without an edit here.
var inputPaths = []string{
	"catalog/snapshot.json", ".github/reconcile-models", ".github/workflows/reconcile-models.yml",
	".claude/skills/reconcile-models", ".claude/skills/add-model",
}

// mediaTypes are what a markdown or html page must be served as, so an error
// page in another format never hashes as content.
var mediaTypes = map[string]string{typeMarkdown: "text/markdown", typeHTML: "text/html"}

// fingerprints is one run's record: when the pages were read, a hash per
// page (or why it couldn't be read) and per input file.
type fingerprints struct {
	Version   int               `json:"version"`
	Provider  string            `json:"provider"`
	Commit    string            `json:"commit"`
	FetchedAt time.Time         `json:"fetched_at"`
	Sources   map[string]source `json:"sources"`
	Inputs    map[string]string `json:"inputs"`
}

// hashInputs hashes every file under the input paths, keyed by its path. A
// path that can't be read is recorded, so it can't match a baseline.
func hashInputs(paths []string) map[string]string {
	inputs := map[string]string{}

	for _, root := range paths {
		err := filepath.WalkDir(root, func(file string, entry fs.DirEntry, err error) error {
			if err != nil || entry.IsDir() {
				return err
			}

			raw, err := os.ReadFile(file)
			if err != nil {
				return err
			}

			sum := sha256.Sum256(raw)
			inputs[filepath.ToSlash(file)] = hex.EncodeToString(sum[:])

			return nil
		})
		if err != nil {
			inputs[root] = "unreadable: " + err.Error()
		}
	}

	return inputs
}

type source struct {
	URL    string `json:"url"`
	SHA256 string `json:"sha256,omitempty"`
	Error  string `json:"error,omitempty"`
}

// fetcher fetches a page and returns its body and media type.
type fetcher func(ctx context.Context, url string) ([]byte, string, error)

// newFetcher fetches through webfetch, which sends its own User-Agent (Google
// serves a 2-byte stub to Go's default one), sets timeouts and blocks private
// addresses. A redirect (webfetch follows none), a status other than 200 or a
// cut-off body is an error.
func newFetcher(opts ...webfetch.Option) fetcher {
	tool := webfetch.New(append([]webfetch.Option{
		webfetch.WithTimeout(60 * time.Second), webfetch.WithMaxBytes(32 << 20), webfetch.WithMaxRedirects(0),
		webfetch.WithAllowedMedia([]string{"text/markdown", "text/html", "application/json", "application/octet-stream"}),
		webfetch.WithFencing(false), webfetch.WithConvertToMarkdown(false),
	}, opts...)...)

	return func(ctx context.Context, url string) ([]byte, string, error) {
		args, err := json.Marshal(map[string]string{"url": url})
		if err != nil {
			return nil, "", err
		}

		raw, err := tool.Execute(ctx, args)
		if err != nil {
			return nil, "", err
		}

		var result struct {
			Error      bool   `json:"error"`
			Message    string `json:"message"`
			StatusCode int    `json:"status_code"`
			MediaType  string `json:"media_type"`
			Truncated  bool   `json:"truncated"`
			Body       string `json:"body"`
		}

		if err := json.Unmarshal(raw, &result); err != nil {
			return nil, "", err
		}

		switch {
		case result.Error:
			return nil, "", fmt.Errorf("fetch: %s", result.Message)
		case result.StatusCode != http.StatusOK:
			return nil, "", fmt.Errorf("status %d", result.StatusCode)
		case result.Truncated:
			return nil, "", errors.New("the body was cut off")
		}

		return []byte(result.Body), result.MediaType, nil
	}
}

// fingerprintPage fetches and hashes one page, returning its stripped text
// for the artifact. A failure is recorded, never hashed, so it can't match.
func fingerprintPage(ctx context.Context, p page, fetch fetcher) (source, string) {
	body, media, err := fetch(ctx, p.URL)
	if err == nil && mediaTypes[p.Type] != "" && media != mediaTypes[p.Type] {
		err = fmt.Errorf("served as %s, want %s", media, mediaTypes[p.Type])
	}

	var text []string
	if err == nil {
		text, err = normalize(p, body)
	}

	if err != nil {
		return source{URL: p.URL, Error: err.Error()}, ""
	}

	return source{URL: p.URL, SHA256: hash(text)}, strings.Join(text, "\n") + "\n"
}

// options are how decide weighs a run against its baseline.
type options struct {
	Mode   string
	Force  bool
	MaxAge time.Duration
	Now    time.Time
}

// decision is the gate's verdict and the ids behind it: this run's pages and
// inputs that changed or couldn't be read. Pages only the baseline had are
// counted, never named, because the baseline comes from an earlier run. Same
// records that every page and input matched the baseline whatever the
// verdict, so the record job can still compare a forced or stale run.
type decision struct {
	Verdict  string   `json:"verdict"`
	RunAgent bool     `json:"run_agent"`
	Mode     string   `json:"mode"`
	Same     bool     `json:"same"`
	IDs      []string `json:"ids,omitempty"`
	Gone     int      `json:"gone,omitempty"`
}

// changedKeys lists, sorted, the keys of now whose value isn't before's.
func changedKeys[V comparable](now, before map[string]V) []string {
	var keys []string

	for _, key := range slices.Sorted(maps.Keys(now)) {
		if value, ok := before[key]; !ok || value != now[key] {
			keys = append(keys, key)
		}
	}

	return keys
}

// decide compares this run with the last successful real run. Anything it
// can't prove unchanged runs the agent; only enforce mode ever skips.
func decide(current fingerprints, baseline *fingerprints, opts options) decision {
	d := decision{Mode: opts.Mode}

	var unreadable []string

	for _, id := range slices.Sorted(maps.Keys(current.Sources)) {
		if current.Sources[id].Error != "" {
			unreadable = append(unreadable, id)
		}
	}

	switch {
	case baseline == nil || baseline.Version != current.Version || baseline.Provider != current.Provider:
		d.Verdict = verdictNoBaseline
	case len(unreadable) > 0:
		d.Verdict, d.IDs = verdictUnreadable, unreadable
	default:
		d.IDs = append(changedKeys(current.Sources, baseline.Sources), changedKeys(current.Inputs, baseline.Inputs)...)

		for id := range baseline.Sources {
			if _, ok := current.Sources[id]; !ok {
				d.Gone++
			}
		}

		d.Same = len(d.IDs) == 0 && d.Gone == 0

		switch {
		case !d.Same:
			d.Verdict = verdictChanged
		case opts.Now.Sub(baseline.FetchedAt) > opts.MaxAge:
			d.Verdict = verdictStale
		default:
			d.Verdict = verdictUnchanged
		}
	}

	if opts.Force {
		d.Verdict = verdictForced
	}

	d.RunAgent = d.Verdict != verdictUnchanged || opts.Mode != modeEnforce

	return d
}

// summary is the gate's part of the public run summary: the verdict, and the
// ids behind it.
func summary(d decision, baselineURL string) string {
	baseline := "the last real run"
	if baselineURL != "" {
		baseline = fmt.Sprintf("[run %s](%s)", path.Base(baselineURL), baselineURL)
	}

	var line string

	switch d.Verdict {
	case verdictForced:
		line = "Forced run, so the gate's answer is ignored."
	case verdictNoBaseline:
		line = "No fingerprints from an earlier real run on this branch, so the agent runs."
	case verdictUnreadable:
		line = "A page couldn't be read, so the agent runs."
	case verdictChanged:
		line = "Changed since " + baseline + ", so the agent runs."
	case verdictStale:
		line = "Nothing changed, but " + baseline + " is too old to rely on, so the agent runs."
	case verdictUnchanged:
		line = "**Skipped the agent:** nothing changed since " + baseline + "."
		if d.RunAgent {
			line = "**Would skip:** nothing changed since " + baseline + ". Shadow mode, so the agent runs anyway."
		}
	}

	text := fmt.Sprintf("### Fingerprint gate (%s mode)\n\n%s\n", d.Mode, line)

	if len(d.IDs) > 0 {
		text += "\n`" + strings.Join(d.IDs, "`, `") + "`\n"
	}

	if d.Gone > 0 {
		noun := "page the baseline had is"
		if d.Gone > 1 {
			noun = "pages the baseline had are"
		}

		text += fmt.Sprintf("\n%d %s no longer listed.\n", d.Gone, noun)
	}

	return text
}
