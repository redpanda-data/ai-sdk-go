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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/providers/fireworks"
)

var (
	contractDiagnostic   = flag.Bool("fireworks-cache-diagnostic", false, "run one explicitly selected nonstream Messages cache diagnostic")
	contractCorrectTools = flag.Bool("fireworks-correct-tools", false, "recapture exactly the seven failed tools cases from the immutable full attempt")
)

type contractRequestOptions struct {
	UserPrompt           string `json:"user_prompt,omitempty"`
	CachePrefixBytes     int    `json:"cache_prefix_bytes,omitempty"`
	CacheSettleMS        int    `json:"cache_settle_ms,omitempty"`
	MaxOutputTokens      int    `json:"max_output_tokens,omitempty"`
	ReplyMaxOutputTokens int    `json:"reply_max_output_tokens,omitempty"`
	ToolResult           string `json:"tool_result,omitempty"`
	ExpectedToolResult   string `json:"expected_tool_result,omitempty"`
	CacheProbeID         string `json:"cache_probe_id,omitempty"`
	CacheRepeatCount     int    `json:"cache_repeat_count,omitempty"`
	AffinityOverride     string `json:"affinity_override,omitempty"`
	CacheIsolation       string `json:"cache_isolation,omitempty"`
	ClaudeSessionID      string `json:"claude_session_id,omitempty"`
	PromptCacheKey       string `json:"prompt_cache_key,omitempty"`
	User                 string `json:"user,omitempty"`
	PerfMetrics          bool   `json:"perf_metrics,omitempty"`
	FormatProof          bool   `json:"format_proof,omitempty"`
	AdaptiveProof        bool   `json:"adaptive_proof,omitempty"`
	ExpectedPlain        string `json:"expected_plain,omitempty"`
	CacheCrossover       bool   `json:"cache_crossover,omitempty"`
}

type contractCaptureProvenance struct {
	TestVersion    int       `json:"test_version"`
	CapturedAt     time.Time `json:"captured_at"`
	SDKRevision    string    `json:"sdk_revision"`
	OriginalFailed bool      `json:"original_failed,omitempty"`
}

type contractSourceReport struct {
	Path string `json:"path"`
	Kind string `json:"kind"`
}

type contractCacheBranch struct {
	Model         string               `json:"model"`
	API           catalog.WireProtocol `json:"api"`
	Stream        bool                 `json:"stream"`
	HitObserved   bool                 `json:"hit_observed"`
	FreshObserved bool                 `json:"fresh_observed"`
}

func contractCachePrompt(c contractCase, size int) string {
	var b strings.Builder
	for n := 0; b.Len() < size; n++ {
		_, _ = fmt.Fprintf(&b, "Record %04d: cobalt lantern meadow %d; river stone compass %d; library index %d preserves distinct observations.\n", n, n*17, n*23, n*31)
	}

	probe := c.key()
	if c.Options.CacheProbeID != "" {
		probe = c.Options.CacheProbeID
	}

	return b.String()[:size] + "\nProbe " + probe + ". Reply with exactly OK."
}

func contractHash(b []byte) string { sum := sha256.Sum256(b); return hex.EncodeToString(sum[:]) }
func contractReadReport(t *testing.T, path string) (contractManifest, contractSourceReport) {
	t.Helper()
	b, err := os.ReadFile(path)
	require.NoError(t, err)
	var report contractManifest
	require.NoError(t, json.Unmarshal(b, &report))

	return report, contractSourceReport{Path: path, Kind: report.RunKind}
}

func contractWriteEvidence(t *testing.T, path string, report contractManifest, budget *contractSpend) {
	t.Helper()

	if budget != nil {
		contractReportBudget(&report, budget)
	}

	b, err := json.MarshalIndent(report, "", "  ")
	require.NoError(t, err)
	contractAtomicWrite(t, path, append(b, '\n'))
}

func contractCaptureRequests(t *testing.T, path string, source contractSourceReport, cases []contractCase, options func(contractCase) contractRequestOptions) contractManifest {
	t.Helper()
	require.True(t, *contractLive && *contractRecord, "correction requires explicit live+record flags")

	key := os.Getenv("FIREWORKS_API_KEY")

	if *contractKeyFile != "" {
		b, err := os.ReadFile(*contractKeyFile)
		require.NoError(t, err)

		key = strings.TrimSpace(string(b))
	}

	require.NotEmpty(t, key, "mandatory live correction requires credentials")
	require.False(t, strings.ContainsAny(key, "\r\n"), "invalid credential format")

	ctx, stop := signal.NotifyContext(t.Context(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	budget := contractOpenBudget(ctx, t)
	revision, err := exec.CommandContext(ctx, "git", "rev-parse", "HEAD").Output()
	require.NoError(t, err)

	report := contractManifest{Version: contractVersion, CapturedAt: time.Now().UTC(), SDKRevision: strings.TrimSpace(string(revision)), Scope: "sdk_direct_upstream_only", RunKind: "partial_targeted_correction", Sources: []contractSourceReport{source}}
	if strings.Contains(path, "cache-v4-") {
		report.RunKind = "partial_cache_comparison"
	}

	client := &http.Client{Timeout: 90 * time.Second, CheckRedirect: func(_ *http.Request, _ []*http.Request) error { return http.ErrUseLastResponse }}

	for _, base := range cases {
		if ctx.Err() != nil {
			t.Error("correction interrupted")
			break
		}

		t.Run(base.key(), func(t *testing.T) {
			c := base
			c.Options = options(base)
			evidence := contractEvidence{Case: base, Options: c.Options, Capture: contractCaptureProvenance{TestVersion: contractVersion, CapturedAt: time.Now().UTC(), SDKRevision: report.SDKRevision}}

			defer func() {
				evidence.Failed = t.Failed() || t.Skipped()
				evidence.NativeVerified = !evidence.Failed
				report.Evidence = append(report.Evidence, evidence)
				contractWriteEvidence(t, path, report, budget)
			}()

			contractExercise(t, c, func(request map[string]any) contractExchange {
				e := contractHTTP(t, client, "https://api.fireworks.ai", key, c, request, budget)
				safe := contractSanitize(t, e, key)
				evidence.Exchanges = append(evidence.Exchanges, safe)
				pending := report
				row := evidence
				row.Failed = true
				pending.Evidence = append(append([]contractEvidence{}, report.Evidence...), row)
				contractWriteEvidence(t, path, pending, budget)

				wireCase := c
				if stream, ok := request["stream"].(bool); ok {
					wireCase.Stream = stream
				}

				budget.reconcile(t, wireCase, e)

				return e
			})
		})
	}

	contractReportBudget(&report, budget)
	t.Logf("PARTIAL correction only: %d cases, known new USD %.9f; cumulative conservative USD %.9f", len(report.Evidence), budget.known, budget.spent+budget.reserved)

	return report
}

func contractCacheDiagnosticOptions(prefix int) contractRequestOptions {
	const session = "7a5c2e10-9b84-4f61-8c33-0d72e906ab51"
	return contractRequestOptions{CachePrefixBytes: prefix, CacheProbeID: "cache-crossover-v5", CacheRepeatCount: 3, CacheSettleMS: 2000, MaxOutputTokens: 16, AffinityOverride: session, ClaudeSessionID: session, CacheIsolation: "sdk-cache-crossover-v5"}
}

//nolint:tparallel // The approved diagnostic has exactly one sequential, budget-reserved case.
func TestFireworksContractCacheDiagnostic(t *testing.T) {
	t.Parallel()

	if !*contractDiagnostic {
		return
	}

	require.Contains(t, []int{8192, 65536}, *contractCachePrefix)
	_, source := contractReadReport(t, "testdata/contracts/full-matrix-v2-incomplete.json")
	c := contractCase{Model: fireworks.ModelDeepSeekV41Flash1017, API: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportDirect, Feature: contractCacheRepeat}
	path := fmt.Sprintf("testdata/contracts/cache-diagnostic-%d.json", *contractCachePrefix)

	report := contractCaptureRequests(t, path, source, []contractCase{c}, func(contractCase) contractRequestOptions {
		return contractCacheDiagnosticOptions(*contractCachePrefix)
	})
	for _, captured := range report.Evidence {
		contractObserveEvidence(t, &captured)
		t.Logf("empirical %d-byte/2s diagnostic: cache_hit_observed=%t fresh_input_observed=%t; not a caching guarantee", *contractCachePrefix, captured.CacheHitObserved, captured.FreshInputObserved)

		for n, exchange := range captured.Exchanges {
			r := contractParse(t, captured.Case, exchange)
			t.Logf("NS CACHE OBSERVATION call=%d input=%d cached=%d output=%d", n+1, r.Input, r.Cached, r.Output)
		}
	}
}

//nolint:tparallel // Paid corrective requests run sequentially with bounded per-run reservations.
func TestFireworksContractToolCorrections(t *testing.T) {
	t.Parallel()

	if !*contractCorrectTools {
		return
	}

	full, source := contractReadReport(t, "testdata/contracts/full-matrix-v2-incomplete.json")
	cases := []contractCase{}

	for _, e := range full.Evidence {
		if e.Failed && e.Case.Feature == contractTools {
			cases = append(cases, e.Case)
		}
	}

	require.Len(t, cases, 7, "only the seven reviewed failed tool cases are authorized")
	contractCaptureRequests(t, "testdata/contracts/tool-corrections.json", source, cases, func(c contractCase) contractRequestOptions {
		opts := contractRequestOptions{UserPrompt: "Call lookup with key x exactly once. Its response is JSON. Reply with the exact value of the result field, without inventing a value.", ToolResult: `{"result":"violet-lantern-zebra"}`, ExpectedToolResult: "violet-lantern-zebra"}
		if c.Model == fireworks.ModelMiniMaxM3 {
			opts.ReplyMaxOutputTokens = 2048
		}

		return opts
	})
}

func contractObserveEvidence(t *testing.T, e *contractEvidence) {
	t.Helper()

	e.CacheHitObserved = false

	e.FreshInputObserved = false
	for _, exchange := range e.Exchanges {
		if exchange.Status != 200 || exchange.ErrorClass != "" || contractControlRejection(e.Case.Feature) {
			continue
		}

		if e.Options.CacheCrossover {
			var body map[string]any
			require.NoError(t, json.Unmarshal(exchange.Request, &body))

			if body["stream"] != e.Case.Stream {
				continue
			}
		}
		var r contractResult

		if e.Case.Feature == contractCancel {
			var final bool

			r, final = contractCancelledUsage(e.Case, exchange)
			if !final {
				continue
			}
		} else {
			r = contractParse(t, e.Case, exchange)
			contractAssertUsage(t, e.Case, r)
			contractAssertPrice(t, e.Case, r)
		}

		fresh := r.Input
		if e.Case.API != catalog.WireProtocolAnthropicMessages {
			fresh -= r.Cached
		}

		e.CacheHitObserved = e.CacheHitObserved || r.Cached > 0
		e.FreshInputObserved = e.FreshInputObserved || fresh > 0
	}
}

func contractCacheGaps(t *testing.T, report *contractManifest, cases []contractCase) []contractCacheBranch {
	t.Helper()

	observed := map[contractCacheBranch]contractCacheBranch{}

	for _, row := range append(append([]contractEvidence{}, report.Evidence...), report.Supplemental...) {
		e := row
		if e.Failed || !e.NativeVerified {
			continue
		}

		contractObserveEvidence(t, &e)
		key := contractCacheBranch{Model: e.Case.Model, API: e.Case.API, Stream: e.Case.Stream}
		value := observed[key]
		value.Model = key.Model
		value.API = key.API
		value.Stream = key.Stream
		value.HitObserved = value.HitObserved || e.CacheHitObserved
		value.FreshObserved = value.FreshObserved || e.FreshInputObserved
		observed[key] = value
	}
	var missing []contractCacheBranch

	for _, c := range cases {
		if c.Feature != contractCacheRepeat {
			continue
		}

		key := contractCacheBranch{Model: c.Model, API: c.API, Stream: c.Stream}

		value, ok := observed[key]
		if !ok {
			value = key
		}

		if !value.HitObserved || !value.FreshObserved {
			missing = append(missing, value)
		}
	}

	return missing
}

func TestFireworksContractNativeReplay(t *testing.T) {
	t.Parallel()

	report := contractLoadManifest(t)
	for _, e := range append(append(append([]contractEvidence{}, report.Evidence...), report.Supplemental...), report.Diagnostics...) {
		t.Run(e.Case.key(), func(t *testing.T) {
			t.Parallel()
			require.False(t, e.Failed, "failed recording is not a regression fixture")
			contractVerifyCrossoverCapture(t, e)
			contractReplay(t, e, 0)
			contractReplay(t, e, 3)
		})
	}
}
