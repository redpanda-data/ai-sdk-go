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
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"go/scanner"
	"go/token"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/catalog/snapshot"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/fireworks"
)

const contractVersion = 5

const (
	contractEffortNone = "none"
	contractSSEDone    = "[DONE]"
)

const (
	contractComplete         = "complete"
	contractCacheRepeat      = "cache_repeat"
	contractCancel           = "cancel"
	contractIncomplete       = "incomplete"
	contractInvalidAuth      = "invalid_auth"
	contractInvalidModel     = "invalid_model"
	contractMalformed        = "malformed"
	contractSystem           = "system"
	contractMultiturn        = "multiturn"
	contractTools            = "tools"
	contractJSON             = "json"
	contractStructured       = "structured"
	contractReasoningHistory = "reasoning_history"
	contractReasoning        = "reasoning"
	contractVision           = "vision"
	contractAdaptive         = "adaptive"
	contractPriority         = "priority"
	contractInvalidFormat    = "json_invalid_format"
	contractInvalidSchema    = "structured_invalid_schema"
	contractInvalidAdaptive  = "adaptive_invalid_mode"
)

var (
	contractLive    = flag.Bool("fireworks-live", false, "run all direct Fireworks contracts; missing credentials and required evidence fail")
	contractRecord  = flag.Bool("fireworks-record", false, "explicitly replace sanitized evidence after a fully successful live run")
	contractKeyFile = flag.String("fireworks-key-file", "", "credential file, read only inside the test process")
	contractBudget  = flag.Float64("fireworks-budget-usd", 0, "explicit per-run conservative request spend limit in USD")
)

type contractCase struct {
	Model     string                 `json:"model"`
	API       catalog.WireProtocol   `json:"api"`
	Transport catalog.WireTransport  `json:"transport"`
	Stream    bool                   `json:"stream"`
	Feature   string                 `json:"feature"`
	Effort    string                 `json:"effort,omitempty"`
	Options   contractRequestOptions `json:"-"`
}

func (c contractCase) key() string {
	sum := sha256.Sum256([]byte(strings.Join([]string{c.Model, string(c.API), string(c.Transport), strconv.FormatBool(c.Stream), c.Feature, c.Effort}, "\x00")))
	return hex.EncodeToString(sum[:12])
}

type contractExchange struct {
	Method              string            `json:"method"`
	Authorization       string            `json:"authorization_class"`
	Path                string            `json:"path"`
	RequestHeaders      map[string]string `json:"request_headers"`
	Request             json.RawMessage   `json:"request"`
	Status              int               `json:"status"`
	ResponseHeaders     map[string]string `json:"response_headers"`
	Response            string            `json:"response"`
	Cancelled           bool              `json:"cancelled,omitempty"`
	RequestWritten      bool              `json:"request_written,omitempty"`
	ResponseTruncated   bool              `json:"response_truncated,omitempty"`
	ErrorClass          string            `json:"error_class,omitempty"`
	ElapsedMS           int64             `json:"elapsed_ms,omitempty"`
	StartedAt           time.Time         `json:"started_at,omitzero"`
	PromptSHA256        string            `json:"prompt_sha256,omitempty"`
	ResponseHeaderNames []string          `json:"response_header_names,omitempty"`
}

type contractEvidence struct {
	Case               contractCase              `json:"case"`
	Failed             bool                      `json:"failed,omitempty"`
	Exchanges          []contractExchange        `json:"exchanges"`
	Options            contractRequestOptions    `json:"request_options,omitzero"`
	Capture            contractCaptureProvenance `json:"capture"`
	NativeVerified     bool                      `json:"native_verified"`
	CacheHitObserved   bool                      `json:"cache_hit_observed"`
	FreshInputObserved bool                      `json:"fresh_input_observed"`
}

type contractManifest struct {
	Version                    int                    `json:"version"`
	CatalogFingerprint         string                 `json:"catalog_fingerprint"`
	TestFingerprint            string                 `json:"test_fingerprint"`
	CapturedAt                 time.Time              `json:"captured_at"`
	SDKRevision                string                 `json:"sdk_revision"`
	Scope                      string                 `json:"scope"`
	Blocked                    []string               `json:"blocked"`
	ObservedUSD                float64                `json:"observed_usd,omitempty"`
	KnownRunUSD                float64                `json:"known_run_usd"`
	PriorConservativeUSD       float64                `json:"prior_conservative_usd"`
	CumulativeConservativeUSD  float64                `json:"cumulative_conservative_usd"`
	UnresolvedReservationsUSD  float64                `json:"unresolved_reservations_usd"`
	Evidence                   []contractEvidence     `json:"evidence"`
	VerifierFingerprint        string                 `json:"verifier_fingerprint,omitempty"`
	RunKind                    string                 `json:"run_kind,omitempty"`
	Sources                    []contractSourceReport `json:"source_reports,omitempty"`
	VerifierCatalogFingerprint string                 `json:"verifier_catalog_fingerprint,omitempty"`
	Superseded                 []contractEvidence     `json:"superseded_evidence,omitempty"`
	MissingCacheBranches       []contractCacheBranch  `json:"missing_cache_branches,omitempty"`
	NativeFailures             []contractCase         `json:"native_failures,omitempty"`
	Supplemental               []contractEvidence     `json:"supplemental_evidence,omitempty"`
	Diagnostics                []contractEvidence     `json:"diagnostic_evidence,omitempty"`
}

func contractInventory(t *testing.T) ([]contractCase, string) {
	t.Helper()
	return contractInventoryAt(t, catalog.Today())
}

func contractInventoryAt(t *testing.T, asOf time.Time) ([]contractCase, string) {
	t.Helper()

	cat := (&fireworks.Provider{}).Catalog()
	require.NotNil(t, cat)
	var cases []contractCase

	for _, o := range cat.All() {
		if cat.At(asOf).IsRetired(o.ID) || (!o.Life.Available.IsZero() && o.Life.Available.After(asOf)) {
			continue
		}

		contractRequireHandledSupport(t, o.Capabilities, o.Reasoning)
		require.NotEmpty(t, o.WireAPIs, "%s has no declared API", o.ID)
		require.True(t, o.Capabilities.Streaming, "%s: stream/nonstream evidence is required", o.ID)
		require.Empty(t, o.Speeds, "%s: speed variants need dedicated evidence", o.ID)
		require.Empty(t, o.Pricing.Default.Brackets, "context brackets require dedicated price/threshold cases")

		for _, override := range o.Pricing.Overrides {
			require.Empty(t, override.RateCard.Brackets, "override context brackets require dedicated cases")
		}

		for _, modality := range o.Modalities.Input {
			require.Contains(t, []catalog.Modality{catalog.ModalityText, catalog.ModalityImage}, modality, "unsupported input modality requires explicit cases")
		}

		for _, modality := range o.Modalities.Output {
			require.Equal(t, catalog.ModalityText, modality, "unsupported output modality requires explicit cases")
		}

		for _, api := range o.WireAPIs {
			support, declared := o.SupportForWireAPI(api)
			require.True(t, declared, "%s: per-API feature declarations are mandatory", o.ID)
			contractRequireHandledSupport(t, support.Capabilities, support.Reasoning)
			require.False(t, support.Capabilities.ToolSearchProtocol != "" || support.Capabilities.Audio, "%s: uncovered per-API capability", o.ID)
			require.False(t, support.Reasoning.Budget, "%s: uncovered manual reasoning budget", o.ID)
			require.False(t, support.Reasoning.ThinkingPrefixCheck, "thinking-prefix preservation requires dedicated cases")
			require.Contains(t, []catalog.WireProtocol{catalog.WireProtocolAnthropicMessages, catalog.WireProtocolOpenAIChatCompletions, catalog.WireProtocolOpenAIResponses}, api.Protocol)
			require.Equal(t, catalog.WireTransportDirect, api.Transport, "only direct upstream capture is supported")

			features := []string{contractComplete, contractIncomplete, contractInvalidAuth, contractInvalidModel, contractMalformed, contractCacheRepeat, contractCancel}
			if support.Capabilities.SystemPrompts {
				features = append(features, contractSystem)
			}

			if support.Capabilities.MultiTurn {
				features = append(features, contractMultiturn)
			}

			if support.Capabilities.Tools {
				features = append(features, contractTools)
			}

			if support.Capabilities.JSONMode {
				features = append(features, contractJSON)
			}

			if support.Capabilities.StructuredOutput {
				features = append(features, contractStructured, contractInvalidSchema)
			}

			if support.Capabilities.Reasoning {
				features = append(features, contractReasoningHistory)
			}

			if support.Capabilities.Vision {
				features = append(features, contractVision)
			}

			if support.Reasoning.Adaptive {
				features = append(features, contractAdaptive)
			}

			for _, override := range o.Pricing.Overrides {
				if override.Match.ServiceTier == llm.ServiceTierPriority {
					if o.Attributes["priority_tier_status"] != "blocked_missing_effective_tier" {
						features = append(features, contractPriority)
					}
				} else {
					require.FailNow(t, "uncovered pricing selector", "%s: %+v", o.ID, override.Match)
				}
			}

			for _, id := range append([]string{o.ID}, o.Aliases...) {
				for _, stream := range []bool{false, true} {
					for _, feature := range features {
						cases = append(cases, contractCase{Model: id, API: api.Protocol, Transport: api.Transport, Stream: stream, Feature: feature})
					}

					for _, effort := range support.Reasoning.Efforts {
						cases = append(cases, contractCase{Model: id, API: api.Protocol, Transport: api.Transport, Stream: stream, Feature: contractReasoning, Effort: string(effort)})
					}
				}
			}
		}
	}

	require.NotEmpty(t, cases, "an empty catalog cannot satisfy contract acceptance")
	var b bytes.Buffer
	require.NoError(t, snapshot.Encode(&b, cat))
	sum := sha256.Sum256(b.Bytes())

	return cases, hex.EncodeToString(sum[:])
}

func contractTestFingerprint(t *testing.T) string {
	t.Helper()

	files, err := filepath.Glob("*contract*_test.go")
	require.NoError(t, err)
	require.NotEmpty(t, files)

	h := sha256.New()

	for _, file := range files {
		b, err := os.ReadFile(file)
		require.NoError(t, err)
		var scan scanner.Scanner
		scan.Init(token.NewFileSet().AddFile(file, -1, len(b)), b, nil, 0)

		for {
			_, tok, literal := scan.Scan()
			if tok == token.EOF {
				break
			}

			_, err = h.Write([]byte(tok.String() + "\x00" + literal + "\x00"))
			require.NoError(t, err)
		}
	}

	rates, err := os.ReadFile("testdata/contracts/published-rates.json")
	require.NoError(t, err)
	_, err = h.Write(rates)
	require.NoError(t, err)

	return hex.EncodeToString(h.Sum(nil))
}

func contractLoadManifest(t *testing.T) contractManifest {
	t.Helper()
	cases, fingerprint := contractInventory(t)

	b, err := os.ReadFile("testdata/contracts/manifest.json")
	if os.IsNotExist(err) {
		b, err = os.ReadFile("testdata/contracts/revalidated-evidence.json")
	}

	require.NoError(t, err, "missing qualified real Fireworks evidence: run explicit capture/revalidation; incomplete attempts do not qualify")
	var manifest contractManifest
	require.NoError(t, json.Unmarshal(b, &manifest))
	require.Equal(t, contractVersion, manifest.Version)
	require.Equal(t, fingerprint, contractCompatibleCatalog(t, manifest), "only explicitly proved SDK-admission narrowing can reuse older captures")

	verified := manifest.VerifierFingerprint
	if verified == "" {
		verified = manifest.TestFingerprint
	}

	require.Equal(t, contractTestFingerprint(t), verified, "verifier drift requires explicit all-capture revalidation; changed requests require fresh live evidence")
	require.Equal(t, "sdk_direct_upstream_only", manifest.Scope)
	require.False(t, manifest.CapturedAt.IsZero())
	require.NotEmpty(t, manifest.SDKRevision)
	capturedCases, _ := contractInventoryAt(t, manifest.CapturedAt)
	capturedKeys := map[string]bool{}
	activeKeys := map[string]bool{}

	for _, c := range capturedCases {
		capturedKeys[c.key()] = true
	}

	for _, c := range cases {
		activeKeys[c.key()] = true
	}

	require.Empty(t, manifest.NativeFailures, "native cases remain unqualified")
	require.Empty(t, manifest.MissingCacheBranches, "exact offering/API/mode positive usage branches remain unqualified")
	require.Empty(t, contractCacheGaps(t, &manifest, cases), "recomputed authoritative final-usage branches remain unqualified")

	byKey := make(map[string]contractEvidence, len(manifest.Evidence))
	for _, evidence := range manifest.Evidence {
		require.True(t, capturedKeys[evidence.Case.key()], "unknown/surplus captured case")

		if !activeKeys[evidence.Case.key()] {
			o, ok := fireworks.Catalog().Resolve(evidence.Case.Model)
			require.True(t, ok, "unknown historical offering")
			require.True(t, fireworks.Catalog().Now().IsRetired(o.ID), "only known retired offerings may contribute historical surplus evidence")
		}

		require.NotEmpty(t, evidence.Capture.TestFingerprint, "missing per-case capture provenance")
		require.NotEmpty(t, evidence.Capture.SDKRevision)
		require.False(t, evidence.Capture.CapturedAt.IsZero())
		require.Len(t, evidence.Capture.RequestSHA256, len(evidence.Exchanges))

		for i, exchange := range evidence.Exchanges {
			require.Equal(t, evidence.Capture.RequestSHA256[i], contractRequestHash(exchange.Request), "request capture fingerprint changed")
		}

		_, duplicate := byKey[evidence.Case.key()]
		require.False(t, duplicate, "duplicate evidence")

		byKey[evidence.Case.key()] = evidence
	}

	for _, c := range cases {
		evidence, ok := byKey[c.key()]
		require.True(t, ok, "missing required case %+v", c)
		require.Equal(t, c, evidence.Case)
		require.False(t, evidence.Failed, "failed or skipped required case")
	}

	return manifest
}

func TestFireworksContractCompleteness(t *testing.T) {
	t.Parallel()
	contractLoadManifest(t)
}

func TestFireworksContractReplay(t *testing.T) {
	t.Parallel()

	manifest := contractLoadManifest(t)
	for _, evidence := range manifest.Evidence {
		t.Run(evidence.Case.key(), func(t *testing.T) {
			t.Parallel()
			contractReplay(t, evidence, 0)
			contractReplay(t, evidence, 3)
		})
	}
}

func contractReplay(t *testing.T, evidence contractEvidence, chunkSize int) {
	t.Helper()
	require.NotEmpty(t, evidence.Exchanges)

	next := 0
	var mu sync.Mutex

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()

		if next >= len(evidence.Exchanges) {
			t.Error("unmatched replay request")
			w.WriteHeader(http.StatusBadRequest)

			return
		}

		e := evidence.Exchanges[next]
		next++

		body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
		if !assert.NoError(t, err) {
			w.WriteHeader(http.StatusBadRequest)
			return
		}

		valid := assert.Equal(t, e.Method, r.Method) && assert.Equal(t, e.Path, r.URL.RequestURI())

		auth := "Bearer replay-only-not-a-credential"
		if e.Authorization == "invalid" {
			auth = "Bearer invalid-contract-probe"
		}

		valid = assert.Equal(t, auth, r.Header.Get("Authorization")) && valid
		for name, value := range e.RequestHeaders {
			valid = assert.Equal(t, value, r.Header.Get(name)) && valid
		}

		valid = assert.Empty(t, r.Header.Get("X-Api-Key")) && assert.Empty(t, r.Header.Get("Cookie")) && valid

		valid = assert.JSONEq(t, string(e.Request), string(body)) && valid
		for name := range r.Header {
			_, declared := e.RequestHeaders[name]
			valid = assert.True(t, declared || name == "Authorization" || name == "User-Agent" || name == "Accept-Encoding" || name == "Content-Length" || name == "Connection", "unexpected replay header %s", name) && valid
		}

		if !valid {
			w.WriteHeader(http.StatusBadRequest)
			return
		}

		if e.Cancelled && e.Status == 0 {
			<-r.Context().Done()
			return
		}

		for name, value := range e.ResponseHeaders {
			w.Header().Set(name, value)
		}

		w.WriteHeader(e.Status)

		if e.Cancelled {
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}

			<-r.Context().Done()

			return
		}

		if chunkSize == 0 {
			_, err = io.WriteString(w, e.Response)
			assert.NoError(t, err)

			return
		}

		for offset := 0; offset < len(e.Response); offset += chunkSize {
			_, err = io.WriteString(w, e.Response[offset:min(offset+chunkSize, len(e.Response))])
			if !assert.NoError(t, err) {
				return
			}

			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
		}
	}))
	defer server.Close()
	var observed []contractExchange

	c := evidence.Case
	c.Options = evidence.Options
	c.Options.CacheSettleMS = 0
	contractExercise(t, c, func(request map[string]any) contractExchange {
		e := contractHTTP(t, server.Client(), server.URL, "replay-only-not-a-credential", c, request, nil)
		observed = append(observed, e)

		return e
	})
	mu.Lock()
	consumed := next
	mu.Unlock()
	require.Equal(t, len(evidence.Exchanges), consumed, "unconsumed replay exchange")

	for i, e := range observed {
		require.Equal(t, evidence.Exchanges[i].Response, e.Response, "response/SSE bytes changed")
	}
}

//nolint:tparallel // Paid requests must remain sequential to enforce one in-flight reservation.
func TestFireworksContractLive(t *testing.T) {
	t.Parallel()

	if !*contractLive {
		require.False(t, *contractRecord, "recording requires -fireworks-live")
		t.Skip("opt-in direct upstream acceptance; invoke the documented mandatory acceptance command")
	}

	require.False(t, testing.Short(), "live acceptance cannot run in short mode")
	require.Greater(t, *contractBudget, float64(0))
	cases, fingerprint := contractInventory(t)
	key := os.Getenv("FIREWORKS_API_KEY")

	if *contractKeyFile != "" {
		b, err := os.ReadFile(*contractKeyFile)
		require.NoError(t, err, "cannot read credential file")

		key = strings.TrimSpace(string(b))
	}

	require.NotEmpty(t, key, "mandatory Fireworks acceptance requires FIREWORKS_API_KEY or -fireworks-key-file")
	require.False(t, strings.ContainsAny(key, "\r\n"), "credential must contain a single value")

	client := &http.Client{Timeout: 90 * time.Second, CheckRedirect: func(_ *http.Request, _ []*http.Request) error { return http.ErrUseLastResponse }}
	manifest := contractManifest{Version: contractVersion, CatalogFingerprint: fingerprint, TestFingerprint: contractTestFingerprint(t), CapturedAt: time.Now().UTC(), Scope: "sdk_direct_upstream_only", Blocked: []string{"gateway interception, enforcement, spending, tenant isolation and real CLI clients are separate cloudv2 acceptance", "Priority requests are unsupported where effective-tier response attribution is unavailable"}}
	revision, err := exec.CommandContext(t.Context(), "git", "rev-parse", "HEAD").Output()
	require.NoError(t, err)

	manifest.SDKRevision = strings.TrimSpace(string(revision))
	manifest.RunKind = "full_matrix"
	manifest.VerifierFingerprint = manifest.TestFingerprint

	ctx, stop := signal.NotifyContext(t.Context(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	budget := contractOpenBudget(ctx, t)
	contractPreflight(t, cases, budget)
	contractCheckpoint(t, manifest, nil, budget)

	for _, base := range cases {
		c := base

		c.Options = contractSemanticOptions(base)
		if contractControlRejection(c.Feature) {
			c.Options.MaxOutputTokens = 16
		}

		if ctx.Err() != nil {
			t.Error("live acceptance interrupted; remaining cases are incomplete")
			break
		}

		if !t.Run(c.key(), func(t *testing.T) {
			// Deliberately sequential: one bounded inference request may be in flight.
			evidence := contractEvidence{Case: base, Options: c.Options, Capture: contractCaptureProvenance{TestVersion: contractVersion, TestFingerprint: manifest.TestFingerprint, CapturedAt: time.Now().UTC(), SDKRevision: manifest.SDKRevision}}

			defer func() {
				evidence.Failed = t.Failed() || t.Skipped()
				evidence.NativeVerified = !evidence.Failed
				manifest.Evidence = append(manifest.Evidence, evidence)
				contractCheckpoint(t, manifest, nil, budget)
			}()

			contractExercise(t, c, func(request map[string]any) contractExchange {
				e := contractHTTP(t, client, "https://api.fireworks.ai", key, c, request, budget)
				safe := contractSanitize(t, e, key)
				evidence.Exchanges = append(evidence.Exchanges, safe)
				evidence.Capture.RequestSHA256 = append(evidence.Capture.RequestSHA256, contractRequestHash(safe.Request))
				contractCheckpoint(t, manifest, &evidence, budget)
				budget.reconcile(t, c, e)
				contractCheckpoint(t, manifest, &evidence, budget)

				return e
			})
		}) {
			t.Logf("required case failed: model=%s api=%s stream=%t feature=%s effort=%s", c.Model, c.API, c.Stream, c.Feature, c.Effort)
		}
	}

	manifest.MissingCacheBranches = contractCacheGaps(t, &manifest, cases)
	for _, evidence := range manifest.Evidence {
		if evidence.Failed {
			manifest.NativeFailures = append(manifest.NativeFailures, evidence.Case)
		}
	}

	if len(manifest.MissingCacheBranches) > 0 {
		t.Errorf("native cache misses are valid, but positive final-usage branches remain unqualified: %+v", manifest.MissingCacheBranches)
	}

	contractReportBudget(&manifest, budget)

	if t.Failed() && *contractRecord {
		b, err := json.MarshalIndent(manifest, "", "  ")
		require.NoError(t, err)
		require.NoError(t, os.MkdirAll("testdata/contracts", 0o750))
		require.NoError(t, os.WriteFile("testdata/contracts/incomplete.json", append(b, '\n'), 0o600))
		t.Logf("INCOMPLETE direct run: %d cases, conservative spent/reserved USD %.6f; success manifest unchanged", len(manifest.Evidence), manifest.CumulativeConservativeUSD)
	}

	require.False(t, t.Failed(), "live contracts failed; evidence has NOT been updated")
	require.Len(t, manifest.Evidence, len(cases), "required case skipped or not executed")

	for _, evidence := range manifest.Evidence {
		require.False(t, evidence.Failed, "required case failed or skipped")
	}

	contractReportBudget(&manifest, budget)
	t.Logf("direct Fireworks acceptance: %d required cases, known new usage estimated USD %.6f, cumulative conservative held USD %.6f; gateway scope excluded", len(cases), budget.known, budget.spent+budget.reserved)

	if *contractRecord {
		b, err := json.MarshalIndent(manifest, "", "  ")
		require.NoError(t, err)
		require.False(t, strings.Contains(string(b), key), "credential leaked into evidence") //nolint:testifylint // Never include the credential in assertion diagnostics.
		require.NoError(t, os.MkdirAll("testdata/contracts", 0o750))
		require.NoError(t, os.WriteFile("testdata/contracts/manifest.json", append(b, '\n'), 0o600))
	}
}

func contractSanitize(t *testing.T, e contractExchange, key string) contractExchange {
	t.Helper()

	for name := range e.ResponseHeaders {
		if contractHeaderKind(name) == "" {
			delete(e.ResponseHeaders, name)
		}
	}

	sanitizeJSON := func(raw string) string {
		var value any
		require.NoError(t, json.Unmarshal([]byte(strings.ReplaceAll(raw, key, "[REDACTED]")), &value))
		contractSanitizeValue(value)
		b, err := json.Marshal(value)
		require.NoError(t, err)

		return string(b)
	}

	e.Request = json.RawMessage(sanitizeJSON(string(e.Request)))
	if strings.Contains(e.ResponseHeaders["Content-Type"], "text/event-stream") {
		lines := strings.Split(e.Response, "\n")
		for i, line := range lines {
			if data, ok := strings.CutPrefix(line, "data:"); ok && strings.TrimSpace(data) != contractSSEDone {
				lines[i] = "data: " + sanitizeJSON(strings.TrimSpace(data))
			}
		}

		e.Response = strings.Join(lines, "\n")
	} else if e.Response != "" {
		e.Response = sanitizeJSON(e.Response)
	}

	return e
}

func contractSanitizeValue(value any) {
	switch value := value.(type) {
	case map[string]any:
		for name, item := range value {
			switch name {
			case "id", "call_id", "tool_use_id", "tool_call_id", "request_id", "item_id", "response_id":
				if text, ok := item.(string); ok {
					sum := sha256.Sum256([]byte(text))
					value[name] = "capture_" + hex.EncodeToString(sum[:12])
				}
			case "created", "created_at", "completed_at":
				value[name] = float64(0)
			default:
				contractSanitizeValue(item)
			}
		}
	case []any:
		for _, item := range value {
			contractSanitizeValue(item)
		}
	}
}

func contractHTTP(t *testing.T, client *http.Client, base, key string, c contractCase, body map[string]any, spend *contractSpend) contractExchange {
	t.Helper()

	started := time.Now().UTC()

	if stream, ok := body["stream"].(bool); ok {
		c.Stream = stream
	}

	exchange := contractHTTPExchange(t, client, base, key, c, body, spend)
	exchange.StartedAt = started
	exchange.ElapsedMS = time.Since(started).Milliseconds()
	exchange.PromptSHA256 = contractPromptHash(body)

	return exchange
}

func contractHTTPExchange(t *testing.T, client *http.Client, base, key string, c contractCase, body map[string]any, spend *contractSpend) contractExchange {
	t.Helper()

	b, err := json.Marshal(body)
	require.NoError(t, err)
	require.LessOrEqual(t, len(b), 72*1024, "bounded contract request exceeded limit")

	path := map[catalog.WireProtocol]string{catalog.WireProtocolAnthropicMessages: "/inference/v1/messages", catalog.WireProtocolOpenAIChatCompletions: "/inference/v1/chat/completions", catalog.WireProtocolOpenAIResponses: "/inference/v1/responses"}[c.API]
	require.NotEmpty(t, path)

	if spend != nil {
		spend.reserve(t, c, b, body)
	}

	ctx := t.Context()
	if spend != nil {
		ctx = spend.ctx
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+path, bytes.NewReader(b))
	require.NoError(t, err)

	auth := key
	if c.Feature == contractInvalidAuth {
		auth = "invalid-contract-probe"
	}

	req.Header.Set("Authorization", "Bearer "+auth)

	headers := map[string]string{"Content-Type": "application/json"}
	if c.Feature == contractCacheRepeat {
		headers["X-Session-Affinity"] = "sdk-contract-" + c.key()
	}

	if c.Options.AffinityOverride != "" {
		headers["X-Session-Affinity"] = c.Options.AffinityOverride
	}

	if c.Options.ClaudeSessionID != "" {
		headers["X-Claude-Code-Session-Id"] = c.Options.ClaudeSessionID
	}

	if c.Options.CacheIsolation != "" {
		headers["X-Prompt-Cache-Isolation-Key"] = c.Options.CacheIsolation
	}

	if c.API == catalog.WireProtocolAnthropicMessages {
		headers["Anthropic-Version"] = "2023-06-01"
	}

	for name, value := range headers {
		req.Header.Set(name, value)
	}

	if c.Feature == contractCancel {
		e := contractAbort(t, client, req, b, headers, key)
		e.Response = strings.ReplaceAll(e.Response, key, "[REDACTED]")

		return e
	}

	authClass := "valid"
	if c.Feature == contractInvalidAuth {
		authClass = "invalid"
	}

	e := contractExchange{Method: http.MethodPost, Authorization: authClass, Path: path, RequestHeaders: headers, Request: b, ResponseHeaders: map[string]string{}}

	resp, err := client.Do(req)
	if err != nil {
		e.ErrorClass = "transport"
		return e
	}
	defer resp.Body.Close()

	e.Status = resp.StatusCode
	e.ResponseHeaders, e.ResponseHeaderNames = contractResponseMetadata(resp.Header, key)
	response, err := io.ReadAll(io.LimitReader(resp.Body, (2<<20)+1))
	redacted := strings.ReplaceAll(string(response), key, "[REDACTED]")

	if err != nil {
		e.ErrorClass = "transport_read"
		e.ResponseTruncated = true
	}

	if len(response) > 2<<20 {
		e.ErrorClass = "response_too_large"
		e.ResponseTruncated = true
	}

	e.Response, _ = contractCancelPrefix([]byte(redacted), resp.Header.Get("Content-Type"))
	if e.Response == "" && redacted != "" {
		e.ErrorClass = "non_json_or_truncated_response"
		e.ResponseTruncated = true
	}

	return e
}
