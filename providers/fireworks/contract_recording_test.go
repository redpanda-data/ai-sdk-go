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
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/providers/fireworks"
)

func TestFireworksContractSyntheticPromotedRecordings(t *testing.T) {
	t.Parallel()

	retained := contractReadReport(t, "testdata/contracts/revalidated-evidence.json")
	base := contractManifest{Version: contractVersion, Evidence: retained.Evidence[:1]}
	added := contractManifest{Version: contractVersion, Evidence: retained.Evidence[1:2], Supplemental: retained.Supplemental[:1], Diagnostics: retained.Diagnostics[:1]}
	root := t.TempDir()
	contractWriteEvidence(t, filepath.Join(root, "revalidated-evidence.json"), base)
	contractWriteEvidence(t, filepath.Join(root, "recordings", "added.json"), added)

	loaded, err := contractLoadManifestAt(root)
	require.NoError(t, err)
	require.Equal(t, append(append([]contractEvidence{}, base.Evidence...), added.Evidence...), loaded.Evidence)
	require.Equal(t, added.Supplemental, loaded.Supplemental)
	require.Equal(t, added.Diagnostics, loaded.Diagnostics)

	for _, row := range append(append(loaded.Evidence, loaded.Supplemental...), loaded.Diagnostics...) {
		t.Run(row.Case.key(), func(t *testing.T) {
			t.Parallel()
			contractReplay(t, row, 0)
			contractReplay(t, row, 3)
		})
	}
}

func TestFireworksContractSyntheticRejectPromotedRecordings(t *testing.T) {
	t.Parallel()

	retained := contractReadReport(t, "testdata/contracts/revalidated-evidence.json")
	failed := retained.Evidence[1]
	failed.Failed = true
	empty := retained.Evidence[1]
	empty.Exchanges = nil

	for _, test := range []struct {
		name   string
		report contractManifest
		raw    string
		want   string
	}{
		{name: "malformed", raw: "{", want: "unexpected end of JSON input"},
		{name: "duplicate", report: contractManifest{Evidence: retained.Evidence[:1]}, want: "duplicate recorded case"},
		{name: "failed", report: contractManifest{Evidence: []contractEvidence{failed}}, want: "failed recording"},
		{name: "failed_supplemental", report: contractManifest{Supplemental: []contractEvidence{failed}}, want: "failed recording"},
		{name: "failed_diagnostic", report: contractManifest{Diagnostics: []contractEvidence{failed}}, want: "failed recording"},
		{name: "empty_exchanges", report: contractManifest{Evidence: []contractEvidence{empty}}, want: "no exchanges"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			root := t.TempDir()
			contractWriteEvidence(t, filepath.Join(root, "manifest.json"), contractManifest{Evidence: retained.Evidence[:1]})

			path := filepath.Join(root, "recordings", "added.json")
			if test.raw != "" {
				contractAtomicWrite(t, path, []byte(test.raw))
			} else {
				contractWriteEvidence(t, path, test.report)
			}

			_, err := contractLoadManifestAt(root)
			require.ErrorContains(t, err, test.want)
		})
	}
}

const contractSyntheticKey = "synthetic-only-not-a-credential"

func contractSyntheticNativeOptions(t *testing.T, baseURL, output string) contractNativeOptions {
	t.Helper()

	model := fireworks.ModelDeepSeekV41Flash1017
	other := fireworks.ModelDeepSeekV41Flash

	return contractNativeOptions{
		Catalog: contractCatalogWithNativeOnlyOffering(t), AsOf: catalog.Today(), Model: model,
		Record: true, Output: output, Budget: 1, Key: contractSyntheticKey,
		BaseURL: baseURL, Client: &http.Client{Timeout: 10 * time.Second}, Revision: "synthetic-revision",
		Cases: []contractCase{
			{Model: model, API: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportDirect, Feature: contractComplete},
			{Model: other, API: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportDirect, Feature: contractComplete},
		},
	}
}

func contractSyntheticNativeServer(t *testing.T, usage bool) (*httptest.Server, func() []string) {
	t.Helper()

	var mu sync.Mutex
	var models []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, err := io.ReadAll(r.Body)
		if !assert.NoError(t, err) {
			w.WriteHeader(http.StatusBadRequest)
			return
		}

		var body map[string]any
		if !assert.NoError(t, json.Unmarshal(b, &body)) {
			w.WriteHeader(http.StatusBadRequest)
			return
		}

		model, ok := body["model"].(string)
		assert.True(t, ok)
		mu.Lock()

		models = append(models, model)
		mu.Unlock()
		assert.Equal(t, "Bearer "+contractSyntheticKey, r.Header.Get("Authorization"))
		assert.Equal(t, "/inference/v1/messages", r.URL.Path)

		response := map[string]any{
			"id": "synthetic-response-id", "type": "message", "role": "assistant", "model": model,
			"content": []any{map[string]any{"type": "text", "text": "OK " + contractSyntheticKey}}, "stop_reason": "end_turn",
		}
		if usage {
			response["usage"] = map[string]any{"input_tokens": 9, "output_tokens": 2}
		}

		w.Header().Set("Content-Type", "application/json")
		assert.NoError(t, json.NewEncoder(w).Encode(response))
	}))
	t.Cleanup(server.Close)

	return server, func() []string {
		mu.Lock()
		defer mu.Unlock()

		return append([]string{}, models...)
	}
}

//nolint:tparallel // Exercise the native runner's sequential budget reservations.
func TestFireworksContractSyntheticNativeRecording(t *testing.T) {
	t.Parallel()

	server, models := contractSyntheticNativeServer(t, true)
	output := filepath.Join(t.TempDir(), "capture.json")
	opts := contractSyntheticNativeOptions(t, server.URL, output)
	require.NoError(t, contractRunNative(t, opts))
	require.Equal(t, []string{opts.Model}, models())

	report := contractReadReport(t, output)
	checkpoint := contractReadReport(t, output+".incomplete.json")
	require.Equal(t, report, checkpoint)
	require.Equal(t, "selected_model", report.RunKind)
	require.Len(t, report.Evidence, 1)
	require.False(t, report.Evidence[0].Failed)
	require.True(t, report.Evidence[0].NativeVerified)
	require.Positive(t, report.KnownRunUSD)
	require.InDelta(t, 0, report.UnresolvedReservationsUSD, 1e-12)

	b, err := os.ReadFile(output)
	require.NoError(t, err)
	require.NotContains(t, string(b), contractSyntheticKey)
	require.Contains(t, string(b), "[REDACTED]")
	require.NotContains(t, string(b), "synthetic-response-id")
	contractReplay(t, report.Evidence[0], 3)
}

func TestFireworksContractSyntheticNativeValidation(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name   string
		change func(*contractNativeOptions)
		want   string
	}{
		{"unknown", func(o *contractNativeOptions) { o.Model = "unknown-model" }, "no active cases"},
		{"retired", func(o *contractNativeOptions) {
			o.Model = fireworks.ModelGPTOSS120B
			o.AsOf = time.Date(2030, time.January, 1, 0, 0, 0, 0, time.UTC)
		}, "no active cases"},
		{"missing_output", func(o *contractNativeOptions) { o.Output = "" }, "explicit output path"},
		{"missing_budget", func(o *contractNativeOptions) { o.Budget = 0 }, "finite positive"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			server, models := contractSyntheticNativeServer(t, true)
			opts := contractSyntheticNativeOptions(t, server.URL, filepath.Join(t.TempDir(), "capture.json"))
			test.change(&opts)
			require.ErrorContains(t, contractRunNative(t, opts), test.want)
			require.Empty(t, models(), "invalid native options must fail before HTTP")
		})
	}
}

func TestFireworksContractSyntheticNativeFailure(t *testing.T) {
	t.Parallel()

	server, models := contractSyntheticNativeServer(t, false)
	output := filepath.Join(t.TempDir(), "capture.json")
	prior := []byte("pre-existing successful recording\n")
	require.NoError(t, os.WriteFile(output, prior, 0o600))

	executable, err := os.Executable()
	require.NoError(t, err)
	cmd := exec.CommandContext(t.Context(), executable, "-test.run=^TestFireworksContractSyntheticNativeFailureWorker$", "-test.v")

	cmd.Env = append(os.Environ(), "FIREWORKS_SYNTHETIC_OUTPUT="+output, "FIREWORKS_SYNTHETIC_URL="+server.URL)
	logs, err := cmd.CombinedOutput()
	require.Error(t, err, "a malformed-usage response must fail the native run: %s", logs)
	require.Contains(t, string(logs), "missing final input usage")
	require.Equal(t, []string{fireworks.ModelDeepSeekV41Flash1017}, models())

	b, err := os.ReadFile(output)
	require.NoError(t, err)
	require.Equal(t, prior, b, "failed run must preserve the requested success file")
	checkpoint := contractReadReport(t, output+".incomplete.json")
	require.Len(t, checkpoint.Evidence, 1)
	require.True(t, checkpoint.Evidence[0].Failed)
	require.False(t, checkpoint.Evidence[0].NativeVerified)
	require.Len(t, checkpoint.NativeFailures, 1)
	require.Positive(t, checkpoint.UnresolvedReservationsUSD)
	require.Zero(t, checkpoint.KnownRunUSD)
	require.Len(t, checkpoint.Evidence[0].Exchanges, 1, "checkpoint must precede parsing final usage")

	b, err = os.ReadFile(output + ".incomplete.json")
	require.NoError(t, err)
	require.NotContains(t, string(b), contractSyntheticKey)
	require.Contains(t, string(b), "[REDACTED]")
}

//nolint:tparallel // Exercise the native runner's sequential budget reservations.
func TestFireworksContractSyntheticNativeFailureWorker(t *testing.T) {
	t.Parallel()

	output := os.Getenv("FIREWORKS_SYNTHETIC_OUTPUT")
	if output == "" {
		t.Skip("subprocess helper for an expected native assertion failure")
	}

	opts := contractSyntheticNativeOptions(t, os.Getenv("FIREWORKS_SYNTHETIC_URL"), output)
	require.ErrorContains(t, contractRunNative(t, opts), "success recording unchanged")
}
