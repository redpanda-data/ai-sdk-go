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

// Package a2ui publishes opt-in, read-only A2UI 1.0 snapshots over A2A 1.0.
// It does not modify the default A2A executor or interpret model output.
package a2ui

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/a2aproject/a2a-go/v2/a2a"
)

const (
	// Version is the supported A2UI wire version.
	Version = "1.0"
	// ExtensionURI identifies the A2UI 1.0 A2A extension.
	ExtensionURI = "https://a2ui.org/a2a-extension/a2ui/v1.0"
	// MIMEType identifies A2UI messages in A2A data part metadata.
	MIMEType = "application/a2ui+json"
	// BasicCatalogID is the catalog used by this read-only snapshot subset.
	BasicCatalogID = "https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json"

	rendererCapabilitiesKey = "a2uiRendererCapabilities"
	maxOfferEntries         = 64
	maxComponents           = 64
	maxDepth                = 8
	maxIDBytes              = 128
	maxTextBytes            = 8 << 10
	maxSnapshotBytes        = 64 << 10
)

// Kind identifies a component in the supported Basic Catalog subset.
type Kind string

const (
	// Text displays a literal string.
	Text Kind = "Text"
	// Row lays out children horizontally.
	Row Kind = "Row"
	// Column lays out children vertically.
	Column Kind = "Column"
)

// Component is one node in a snapshot. Text uses only Text; Row and Column use
// only Children, an ordered list of IDs in the same snapshot.
type Component struct {
	ID       string
	Kind     Kind
	Text     string
	Children []string
}

// Snapshot creates one read-only surface. SurfaceID must be unique for the
// renderer's lifetime. Its components form one tree rooted at ID "root".
type Snapshot struct {
	SurfaceID  string
	Components []Component
}

var errInvalidSnapshot = errors.New("invalid A2UI snapshot")

// AgentExtension returns an optional agent-card advertisement. Add it only to
// A2A 1.0 endpoints that can respond with snapshots.
func AgentExtension() *a2a.AgentExtension {
	return &a2a.AgentExtension{
		URI:         ExtensionURI,
		Description: "Read-only A2UI 1.0 Text, Row and Column snapshots",
		Params:      map[string]any{"supportedCatalogIds": []any{BasicCatalogID}},
	}
}

// SupportsSnapshot reports whether this message offers the supported version
// and catalog. An extension header alone is not an offer. Use this before
// asking a model for UI or constructing a snapshot.
func SupportsSnapshot(message *a2a.Message) bool {
	if message == nil {
		return false
	}

	if caps, ok := message.Metadata[rendererCapabilitiesKey]; ok {
		return supportsCatalog(caps)
	}

	if len(message.Parts) > maxOfferEntries {
		return false
	}

	for _, part := range message.Parts {
		if part == nil {
			continue
		}

		if _, ok := part.Content.(a2a.Data); !ok {
			continue
		}

		if caps, ok := part.Metadata[rendererCapabilitiesKey]; ok {
			return supportsCatalog(caps)
		}
	}

	return false
}

func supportsCatalog(caps any) bool {
	versions, ok := caps.(map[string]any)
	if !ok {
		return false
	}

	current, ok := versions[Version].(map[string]any)
	if !ok {
		return false
	}

	switch ids := current["supportedCatalogIds"].(type) {
	case []string:
		return len(ids) <= maxOfferEntries && slices.Contains(ids, BasicCatalogID)
	case []any:
		if len(ids) > maxOfferEntries {
			return false
		}
		var supported bool

		for _, value := range ids {
			id, ok := value.(string)
			if !ok {
				return false
			}

			supported = supported || id == BasicCatalogID
		}

		return supported
	default:
		return false
	}
}

// SnapshotPart returns a validated A2UI data part for the current message,
// or (nil, nil) if its renderer has not offered the supported version and
// catalog. Keep a plain-text reply alongside the optional part. The snapshot
// is validated only when this turn can receive it.
func SnapshotPart(message *a2a.Message, snapshot Snapshot) (*a2a.Part, error) {
	if !SupportsSnapshot(message) {
		return nil, nil //nolint:nilnil // Negotiation declined; the normal text reply is not an error.
	}

	components, err := snapshot.wireComponents()
	if err != nil {
		return nil, err
	}

	payload := []any{map[string]any{
		"version": Version,
		"createSurface": map[string]any{
			"surfaceId":  snapshot.SurfaceID,
			"catalogId":  BasicCatalogID,
			"components": components,
		},
	}}
	// All payload members have bounded input; now cap the encoded representation,
	// including JSON escaping and the overhead from the message envelope.
	encoded, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("encode A2UI snapshot: %w", err)
	}

	if len(encoded) > maxSnapshotBytes {
		return nil, fmt.Errorf("%w: encoded snapshot exceeds %d bytes", errInvalidSnapshot, maxSnapshotBytes)
	}

	part := a2a.NewDataPart(payload)
	part.Metadata = map[string]any{"mimeType": MIMEType}

	return part, nil
}

func (snapshot Snapshot) wireComponents() ([]any, error) {
	if !validID(snapshot.SurfaceID) || len(snapshot.Components) == 0 || len(snapshot.Components) > maxComponents {
		return nil, fmt.Errorf("%w: invalid surface ID or component count", errInvalidSnapshot)
	}

	components := make(map[string]Component, len(snapshot.Components))
	for i, node := range snapshot.Components {
		if !validID(node.ID) {
			return nil, fmt.Errorf("%w: component %d has an invalid ID", errInvalidSnapshot, i)
		}

		if _, exists := components[node.ID]; exists {
			return nil, fmt.Errorf("%w: component %d has a duplicate ID", errInvalidSnapshot, i)
		}

		if !validComponent(node) {
			return nil, fmt.Errorf("%w: component %d has an unsupported kind or invalid properties", errInvalidSnapshot, i)
		}

		components[node.ID] = node
	}

	if err := validateTree(components); err != nil {
		return nil, err
	}

	wire := make([]any, 0, len(snapshot.Components))
	for _, node := range snapshot.Components {
		part := map[string]any{"id": node.ID, "component": string(node.Kind)}
		if node.Kind == Text {
			part["text"] = node.Text
		} else {
			children := make([]any, len(node.Children))
			for i, id := range node.Children {
				children[i] = id
			}

			part["children"] = children
		}

		wire = append(wire, part)
	}

	return wire, nil
}

func validID(id string) bool {
	return len(id) <= maxIDBytes && strings.TrimSpace(id) != "" && utf8.ValidString(id)
}

func validComponent(node Component) bool {
	switch node.Kind {
	case Text:
		return len(node.Children) == 0 && len(node.Text) <= maxTextBytes && utf8.ValidString(node.Text)
	case Row, Column:
		if node.Text != "" || len(node.Children) == 0 || len(node.Children) > maxComponents {
			return false
		}

		for _, id := range node.Children {
			if !validID(id) {
				return false
			}
		}

		return true
	default:
		return false
	}
}

func validateTree(components map[string]Component) error {
	visited := make(map[string]bool, len(components))
	var visit func(string, int) bool

	visit = func(id string, depth int) bool {
		node, exists := components[id]
		if depth > maxDepth || !exists || visited[id] {
			return false
		}

		visited[id] = true

		for _, child := range node.Children {
			if !visit(child, depth+1) {
				return false
			}
		}

		return true
	}
	if !visit("root", 1) || len(visited) != len(components) {
		return fmt.Errorf("%w: components must form one tree rooted at root with depth at most %d", errInvalidSnapshot, maxDepth)
	}

	return nil
}
