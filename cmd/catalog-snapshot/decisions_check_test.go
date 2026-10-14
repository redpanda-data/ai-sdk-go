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
	"bytes"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/bedrock"
	"github.com/redpanda-data/ai-sdk-go/providers/google"
	"github.com/redpanda-data/ai-sdk-go/providers/meta"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/vertex"
)

// decision is one entry in .github/reconcile-models/decisions.yaml (RULES.md D1).
type decision struct {
	ID       string `yaml:"id"`
	Offering string `yaml:"offering"`
	Field    string `yaml:"field"`
	Kind     string `yaml:"kind"`
	Lock     any    `yaml:"lock"`
	Reason   string `yaml:"reason"`
	Decided  string `yaml:"decided"`
	ReviewBy string `yaml:"review_by"`
	PR       int    `yaml:"pr"`
}

// The two kinds of decision, and the field a hold applies to.
const (
	kindKeep      = "keep"
	kindHold      = "hold"
	fieldNewModel = "new_model"
)

// decisionProviders maps an offering's package prefix to its snapshot provider.
var decisionProviders = map[string]string{
	"anthropic": anthropic.ProviderName,
	"bedrock":   bedrock.ProviderName,
	"google":    google.ProviderName,
	"meta":      meta.ProviderName,
	"openai":    openai.ProviderName,
	"vertex":    vertex.ProviderName,
}

// lockedKeys lists, for each field in RULES.md's Field table, the snapshot
// keys a lock covers: "key" or "key.subkey" on the offering, or "facts.key"
// on its model. A field with one key locks that value; one with several locks
// an object of those present, keyed by their last part.
var lockedKeys = map[string][]string{
	fieldNewModel:      nil,
	"price":            {"pricing"},
	"limit":            {"constraints.max_input_tokens", "constraints.max_output_tokens"},
	"capability":       {"capabilities", "modalities", "reasoning", "speeds"},
	"lifecycle":        {"lifecycle.stage", "lifecycle.available", "lifecycle.deprecated", "lifecycle.retires"},
	"replacement":      {"lifecycle.replaced_by"},
	"release_date":     {"facts.released"},
	"knowledge_cutoff": {"facts.knowledge"},
}

var decisionID = regexp.MustCompile(`^DEC-[0-9]{3}$`)

// snapshotDoc is the part of catalog/snapshot.json a decision can lock.
type snapshotDoc struct {
	Facts     map[string]map[string]any `json:"facts"`
	Providers []struct {
		Provider  string           `json:"provider"`
		Offerings []map[string]any `json:"offerings"`
	} `json:"providers"`
}

// checkDecisions returns one problem per decision that is malformed or that
// the snapshot contradicts.
func checkDecisions(file, snapshotJSON []byte) []string {
	var doc struct {
		Decisions []decision `yaml:"decisions"`
	}

	decoder := yaml.NewDecoder(bytes.NewReader(file))
	decoder.KnownFields(true)

	if err := decoder.Decode(&doc); err != nil {
		return []string{"decisions.yaml: " + err.Error()}
	}

	// json.Number keeps the snapshot's numbers exactly as written.
	var snap snapshotDoc

	jsonDecoder := json.NewDecoder(bytes.NewReader(snapshotJSON))
	jsonDecoder.UseNumber()

	if err := jsonDecoder.Decode(&snap); err != nil {
		return []string{"snapshot: " + err.Error()}
	}

	offerings := map[string]map[string]any{}

	for _, provider := range snap.Providers {
		for _, offering := range provider.Offerings {
			id, _ := offering["id"].(string)
			offerings[provider.Provider+"/"+id] = offering
		}
	}

	var problems []string

	ids, keys := map[string]bool{}, map[string]string{}

	for _, d := range doc.Decisions {
		key := d.Offering + " " + d.Field

		switch problem := d.malformed(); {
		case problem != "":
			problems = append(problems, d.ID+": "+problem)
			continue
		case ids[d.ID]:
			problems = append(problems, d.ID+": duplicate ID")
			continue
		case keys[key] != "":
			problems = append(problems, d.ID+": "+keys[key]+" already covers "+key)
			continue
		}

		ids[d.ID], keys[key] = true, d.ID

		pkg, id, _ := strings.Cut(d.Offering, ":")
		offering, exists := offerings[decisionProviders[pkg]+"/"+id]
		where := fmt.Sprintf("%s %s %s %s: ", d.ID, d.Kind, d.Offering, d.Field)

		switch {
		case d.Kind == kindHold && exists:
			problems = append(problems, where+"the offering is in the catalogue")
		case d.Kind == kindKeep && !exists:
			problems = append(problems, where+"the offering is missing")
		case d.Lock != nil:
			if got, want := canonical(lockedPart(d.Field, offering, snap.Facts)), canonical(d.Lock); got != want {
				problems = append(problems, where+"got "+got+", locked "+want)
			}
		}
	}

	return problems
}

func (d decision) malformed() string {
	pkg, id, _ := strings.Cut(d.Offering, ":")
	_, knownField := lockedKeys[d.Field]
	_, decidedErr := catalog.ParseDate(d.Decided)
	_, reviewErr := catalog.ParseDate(d.ReviewBy)

	switch {
	case !decisionID.MatchString(d.ID):
		return "an ID is DEC- and three digits"
	case d.Kind != kindKeep && d.Kind != kindHold:
		return "kind is keep or hold"
	case decisionProviders[pkg] == "":
		return "unknown package " + pkg
	case id == "":
		return "an offering is <package>:<model ID>"
	case !knownField:
		return "unknown field " + d.Field
	case d.Kind == kindHold && d.Field != fieldNewModel:
		return "a hold applies only to new_model"
	case d.Field == fieldNewModel && d.Lock != nil:
		return "new_model takes no lock"
	case d.Field != fieldNewModel && d.Lock == nil:
		return "a keep needs a lock"
	case d.Reason == "":
		return "a decision needs a reason"
	case decidedErr != nil || reviewErr != nil:
		return "decided and review_by are YYYY-MM-DD dates"
	}

	return ""
}

// lockedPart returns the part of the snapshot a field locks (see lockedKeys).
func lockedPart(field string, offering map[string]any, facts map[string]map[string]any) any {
	model, _ := offering["model"].(string)
	lookup := func(path string) (any, bool) {
		first, rest, nested := strings.Cut(path, ".")
		if first == "facts" {
			value, ok := facts[model][rest]
			return value, ok
		}

		value, ok := offering[first]
		if nested {
			object, _ := value.(map[string]any)
			value, ok = object[rest]
		}

		return value, ok
	}

	paths := lockedKeys[field]
	if len(paths) == 1 {
		value, _ := lookup(paths[0])
		return value
	}

	part := map[string]any{}

	for _, path := range paths {
		if value, ok := lookup(path); ok {
			part[path[strings.LastIndex(path, ".")+1:]] = value
		}
	}

	return part
}

// canonical is a value's JSON with sorted keys, so a lock written in YAML and
// a value decoded from the snapshot compare equal when they hold the same data.
func canonical(value any) string {
	raw, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprintf("<%T that isn't JSON>", value)
	}

	return string(raw)
}
