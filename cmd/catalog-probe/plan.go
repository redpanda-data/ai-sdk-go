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
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/internal/reconcile"
)

const outcomeSkipped = "skipped"

// Probe is one offering to call on its host, and how the call went.
type Probe struct {
	// Provider is the offering's provider package, as reports name it.
	Provider  string `json:"provider"`
	ID        string `json:"id"`
	Region    string `json:"region,omitempty"`
	Outcome   string `json:"outcome"`
	Detail    string `json:"detail,omitempty"`
	RequestID string `json:"request_id,omitempty"`

	// call is the request to send; empty when the plan skips the offering.
	call string
}

// calls are the requests this command can send, by the wire API an
// offering lists, most preferred first.
var calls = []struct{ pkg, wire, call string }{
	{"anthropic", "anthropic_messages/direct", "anthropic"},
	{"openai", "openai_responses/direct", "openai_responses"},
	{"openai", "openai_chat_completions/direct", "openai_chat"},
	{"google", "gemini_generate_content/direct", "gemini"},
	{"bedrock", "bedrock_converse/bedrock_runtime", "bedrock"},
}

// noCredentials names the hosts CI has no credentials for, so their
// offerings are skipped whatever they list.
var noCredentials = map[string]string{
	"meta":   "no Meta credentials in CI yet",
	"vertex": "no Vertex credentials in CI yet",
}

// geoRegions is where a Bedrock geo profile is called from: a source region
// every profile of that geography accepts. A bare ID is called in-region
// from bareRegion, the region CI's tests use.
var geoRegions = map[string]string{
	"us": "us-east-1",
	"eu": "eu-central-1",
	"au": "ap-southeast-2",
	"jp": "ap-northeast-1",
}

const bareRegion = "us-east-1"

// maxCalls caps a run's calls, so a patch listing many offerings can spend
// only a few cents of the keys' budget.
const maxCalls = 50

// plan lists the probes for the vendor legs' folders: each offering a
// folder's snapshot.json adds or changes against its base-snapshot.json,
// both written by pr-job.sh after its checks. Only data is read here, never
// the vendors' code.
func plan(dirs []string) ([]Probe, error) {
	seen := map[string]bool{}

	var probes []Probe

	for _, dir := range dirs {
		base, err := readSnapshot(filepath.Join(dir, "base-snapshot.json"))
		if err != nil {
			return nil, err
		}

		patched, err := readSnapshot(filepath.Join(dir, "snapshot.json"))
		if err != nil {
			return nil, err
		}

		for _, pkg := range slices.Sorted(maps.Keys(reconcile.PackageProviders)) {
			for _, o := range patched.Offerings(pkg) {
				id, _ := o["id"].(string)
				if before, ok := base.Offering(pkg, id); ok && reflect.DeepEqual(reconcile.WithoutDerived(before), reconcile.WithoutDerived(o)) {
					continue
				}

				if key := pkg + " " + id; !seen[key] {
					seen[key] = true

					probes = append(probes, route(Probe{Provider: pkg, ID: id}, o))
				}
			}
		}
	}

	slices.SortFunc(probes, func(a, b Probe) int { return strings.Compare(a.Provider+" "+a.ID, b.Provider+" "+b.ID) })

	called := 0

	for i, p := range probes {
		if p.call == "" {
			continue
		}

		if called++; called > maxCalls {
			probes[i] = skip(p, fmt.Sprintf("over the run's cap of %d calls", maxCalls))
		}
	}

	return probes, nil
}

// route picks the request for an offering from the wire APIs it lists, and
// for Bedrock the region to call from, or skips it with the reason. The ID
// goes into a URL and the PR body, so one the report's offering pattern
// wouldn't accept is dropped.
func route(p Probe, o map[string]any) Probe {
	if !reconcile.OfferingPattern.MatchString(p.ID) {
		p.ID = ""

		return skip(p, "its ID doesn't fit the offering pattern, so it isn't shown or called")
	}

	if reason, ok := noCredentials[p.Provider]; ok {
		return skip(p, reason)
	}

	if p.Provider == "bedrock" {
		prefix, _, _ := strings.Cut(p.ID, ".")

		switch _, profile := reconcile.PrefixNames[prefix]; {
		case prefix == "global":
			return skip(p, "CI's AWS account can't call global profiles")
		case !profile:
			p.Region = bareRegion
		case geoRegions[prefix] == "":
			return skip(p, "no source region for "+prefix+" profiles yet")
		default:
			p.Region = geoRegions[prefix]
		}
	}

	wires, _ := o["wire_apis"].([]any)
	for _, c := range calls {
		if c.pkg == p.Provider && slices.ContainsFunc(wires, func(w any) bool { return wireName(w) == c.wire }) {
			p.call = c.call

			return p
		}
	}

	return skip(p, "no request for its wire APIs yet")
}

func skip(p Probe, reason string) Probe {
	p.Region, p.Outcome, p.Detail, p.call = "", outcomeSkipped, reason, ""

	return p
}

// wireName is a wire API as protocol/transport.
func wireName(w any) string {
	m, _ := w.(map[string]any)

	return fmt.Sprintf("%v/%v", m["protocol"], m["transport"])
}

func readSnapshot(path string) (reconcile.Snapshot, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return reconcile.Snapshot{}, err
	}

	return reconcile.ParseSnapshot(raw)
}
