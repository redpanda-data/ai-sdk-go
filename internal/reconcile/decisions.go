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

// Package reconcile reads the reconcile-models workflow's reviewed inputs.
package reconcile

import (
	"bytes"
	"errors"
	"io"

	"gopkg.in/yaml.v3"
)

// DecisionsPath is decisions.yaml in the repo: the reviewed answers a
// reconcile run must not reopen (RULES.md, Principles).
const DecisionsPath = ".github/reconcile-models/decisions.yaml"

// The two kinds of decision, and the two fields about an offering the
// catalogue doesn't carry yet: a new model, and another copy of one it has.
const (
	KindKeep      = "keep"
	KindHold      = "hold"
	FieldNewModel = "new_model"
	FieldNewCopy  = "new_copy"
)

// Decision is one entry in decisions.yaml. A keep pins one field of an offering to Lock,
// written in that field's catalog/snapshot.json shape, or only the parts Keys
// names; a hold keeps an offering out of the catalogue, or on another field
// leaves the vendor's value out while Rule still asks about it. With a Rule,
// a decision answers only that rule's questions.
type Decision struct {
	ID       string   `yaml:"id"`
	Offering string   `yaml:"offering"`
	Field    string   `yaml:"field"`
	Kind     string   `yaml:"kind"`
	Rule     string   `yaml:"rule"`
	Keys     []string `yaml:"keys"`
	Lock     any      `yaml:"lock"`
	Reason   string   `yaml:"reason"`
	Decided  string   `yaml:"decided"`
	ReviewBy string   `yaml:"review_by"`
	PR       int      `yaml:"pr"`

	// Line is where the entry starts in the file, so a PR can link to it.
	Line int `yaml:"-"`
}

// Answers reports whether d answers a question raised by rule on its
// offering and field: every one when d names no rule, else its rule's.
func (d Decision) Answers(rule string) bool {
	return d.Rule == "" || d.Rule == rule
}

// ParseDecisions decodes decisions.yaml strictly, so an unknown key is an error.
func ParseDecisions(data []byte) ([]Decision, error) {
	var doc struct {
		Decisions []Decision `yaml:"decisions"`
	}

	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)

	if err := decoder.Decode(&doc); err != nil {
		if errors.Is(err, io.EOF) {
			return nil, nil
		}

		return nil, err
	}

	// The node tree has each entry's line; the strict decode above doesn't.
	var root yaml.Node
	if err := yaml.Unmarshal(data, &root); err != nil {
		return nil, err
	}

	mapping := root.Content[0]
	for i := 0; i+1 < len(mapping.Content); i += 2 {
		if mapping.Content[i].Value == "decisions" {
			for j, entry := range mapping.Content[i+1].Content {
				doc.Decisions[j].Line = entry.Line
			}
		}
	}

	return doc.Decisions, nil
}
