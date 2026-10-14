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
// reconcile run must not reopen (RULES.md section D).
const DecisionsPath = ".github/reconcile-models/decisions.yaml"

// The two kinds of decision, and the field a hold applies to.
const (
	KindKeep      = "keep"
	KindHold      = "hold"
	FieldNewModel = "new_model"
)

// Decision is one entry in decisions.yaml. A keep pins one field of an offering to Lock,
// written in that field's catalog/snapshot.json shape; a hold keeps an
// offering out of the catalogue.
type Decision struct {
	ID       string `yaml:"id"`
	Offering string `yaml:"offering"`
	Field    string `yaml:"field"`
	Kind     string `yaml:"kind"`
	Lock     any    `yaml:"lock"`
	Reason   string `yaml:"reason"`
	Decided  string `yaml:"decided"`
	ReviewBy string `yaml:"review_by"`
	PR       int    `yaml:"pr"`

	// Line is where the entry starts in the file, so a PR can link to it.
	Line int `yaml:"-"`
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
