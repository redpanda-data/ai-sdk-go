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
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/internal/reconcile"
)

// Probe is one offering to call on its host, and how the call went.
type Probe struct {
	// Provider is the offering's provider package, as reports name it.
	Provider  string `json:"provider"`
	ID        string `json:"id"`
	Region    string `json:"region,omitempty"`
	Outcome   string `json:"outcome"`
	Detail    string `json:"detail,omitempty"`
	RequestID string `json:"request_id,omitempty"`

	// wire is the wire API the request uses; zero when the offering isn't
	// called.
	wire catalog.WireAPI
}

// calls are the wire APIs this command has a request for, by provider
// package, most preferred first. A package missing here has no CI
// credentials yet, so its offerings are skipped.
var calls = map[string][]catalog.WireAPI{
	"anthropic": {{Protocol: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportDirect}},
	"openai": {
		{Protocol: catalog.WireProtocolOpenAIResponses, Transport: catalog.WireTransportDirect},
		{Protocol: catalog.WireProtocolOpenAIChatCompletions, Transport: catalog.WireTransportDirect},
	},
	"google":  {{Protocol: catalog.WireProtocolGeminiGenerateContent, Transport: catalog.WireTransportDirect}},
	"bedrock": {{Protocol: catalog.WireProtocolBedrockConverse, Transport: catalog.WireTransportBedrockRuntime}},
}

// noRequest is the reason an offering with none of its package's wire APIs
// isn't called.
const noRequest = "no request for its wire APIs yet"

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

// maxProbes caps the offerings a run checks, so a patch listing many can
// spend only a few cents of the keys' budget: each costs at most two model
// calls, the second only when the first didn't work.
const maxProbes = 50

// plan lists the probes for the folders, in the order their changes apply:
// each vendor leg's, then the carry job's for people's commits. Each holds
// its catalogue before and after (base-snapshot.json, snapshot.json), and
// the probes are the offerings a folder adds or changes. The last folder's
// catalogue is the PR's, so an offering a later folder removes isn't called,
// and the rest are routed as they end up. Only data is read here, never the
// commits' code.
func plan(dirs []string) ([]Probe, error) {
	type change struct{ base, patched reconcile.Snapshot }

	changes := make([]change, 0, len(dirs))

	for _, dir := range dirs {
		base, err := readSnapshot(filepath.Join(dir, "base-snapshot.json"))
		if err != nil {
			return nil, err
		}

		patched, err := readSnapshot(filepath.Join(dir, "snapshot.json"))
		if err != nil {
			return nil, err
		}

		changes = append(changes, change{base, patched})
	}

	if len(changes) == 0 {
		return nil, nil
	}

	final := changes[len(changes)-1].patched
	seen := map[string]bool{}

	var probes []Probe

	for _, c := range changes {
		for pkg := range reconcile.PackageProviders {
			for _, o := range c.patched.Offerings(pkg) {
				id, _ := o["id"].(string)
				if before, ok := c.base.Offering(pkg, id); ok && reflect.DeepEqual(reconcile.WithoutDerived(before), reconcile.WithoutDerived(o)) {
					continue
				}

				if latest, kept := final.Offering(pkg, id); kept && !seen[pkg+" "+id] {
					seen[pkg+" "+id] = true

					probes = append(probes, route(Probe{Provider: pkg, ID: id}, latest))
				}
			}
		}
	}

	slices.SortFunc(probes, func(a, b Probe) int { return strings.Compare(a.Provider+" "+a.ID, b.Provider+" "+b.ID) })

	called := 0

	for i, p := range probes {
		if p.wire == (catalog.WireAPI{}) {
			continue
		}

		if called++; called > maxProbes {
			probes[i] = skip(p, fmt.Sprintf("over the run's cap of %d offerings", maxProbes))
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

	wires, ok := calls[p.Provider]
	if !ok {
		return skip(p, "no CI credentials for this host yet")
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

	listed, _ := o["wire_apis"].([]any)
	for _, wire := range wires {
		if slices.ContainsFunc(listed, func(w any) bool { return wireAPI(w) == wire }) {
			p.wire = wire

			return p
		}
	}

	return skip(p, noRequest)
}

func skip(p Probe, reason string) Probe {
	p.Region, p.Outcome, p.Detail, p.wire = "", outcomeSkipped, reason, catalog.WireAPI{}

	return p
}

// wireAPI reads one of a snapshot offering's wire APIs.
func wireAPI(w any) catalog.WireAPI {
	m, _ := w.(map[string]any)
	protocol, _ := m["protocol"].(string)
	transport, _ := m["transport"].(string)

	return catalog.WireAPI{Protocol: catalog.WireProtocol(protocol), Transport: catalog.WireTransport(transport)}
}

func readSnapshot(path string) (reconcile.Snapshot, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return reconcile.Snapshot{}, err
	}

	return reconcile.ParseSnapshot(raw)
}
