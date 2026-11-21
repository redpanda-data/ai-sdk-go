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

package a2ui

import (
	"fmt"
	"math"
	"net/url"
	"slices"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/a2aproject/a2a-go/v2/a2a"
)

const (
	fieldChild = "child"
	fieldURL   = "url"
	fieldLabel = "label"
	fieldValue = "value"

	maxBasicGroupItems = 16
	maxBasicLabelBytes = 256
)

// BasicComponent is a typed literal node in an A2UI 1.0 Basic Catalog surface.
// Construct it with a New function; the zero value is invalid.
type BasicComponent struct {
	id       string
	kind     string
	children []string
	fields   func() map[string]any
}

// BasicSnapshot creates one Basic Catalog surface rooted at "root". It has no
// bindings, functions or model updates. A Button can declare an event, but
// this adapter does not receive events or execute actions.
type BasicSnapshot struct {
	SurfaceID  string
	Components []BasicComponent
}

// BasicSnapshotPart encodes a typed snapshot for this turn's A2UI 1.0 offer.
// It returns (nil, nil) on no offer, just like SnapshotPart. Keep a plain-text
// reply for clients without an offer or when snapshot validation fails.
func BasicSnapshotPart(message *a2a.Message, snapshot BasicSnapshot) (*a2a.Part, error) {
	if !SupportsSnapshot(message) {
		return nil, nil //nolint:nilnil // Negotiation declined; the normal text reply is not an error.
	}

	if !validID(snapshot.SurfaceID) || len(snapshot.Components) == 0 || len(snapshot.Components) > maxComponents {
		return nil, fmt.Errorf("%w: invalid surface ID or component count", errInvalidSnapshot)
	}

	graph := make(map[string][]string, len(snapshot.Components))
	for i, node := range snapshot.Components {
		if node.fields == nil {
			return nil, fmt.Errorf("%w: component %d has invalid properties", errInvalidSnapshot, i)
		}

		if _, exists := graph[node.id]; exists {
			return nil, fmt.Errorf("%w: component %d has a duplicate ID", errInvalidSnapshot, i)
		}

		graph[node.id] = node.children
	}

	if err := validateTree(graph); err != nil {
		return nil, err
	}

	wire := make([]any, 0, len(snapshot.Components))
	for _, node := range snapshot.Components {
		fields := node.fields()
		fields["id"] = node.id
		fields["component"] = node.kind
		wire = append(wire, fields)
	}

	return snapshotPart(snapshot.SurfaceID, wire)
}

func invalidBasicComponent() BasicComponent {
	var node BasicComponent

	return node
}

func basicNode(id, kind string, valid bool, children []string, fields func() map[string]any) BasicComponent {
	if !valid || !validID(id) || len(children) > maxComponents {
		return invalidBasicComponent()
	}

	for _, child := range children {
		if !validID(child) {
			return invalidBasicComponent()
		}
	}

	return BasicComponent{id: id, kind: kind, children: children, fields: fields}
}

func literalText(value string) bool {
	return len(value) <= maxTextBytes && utf8.ValidString(value)
}

func literalLabel(value string) bool {
	return len(value) <= maxBasicLabelBytes && utf8.ValidString(value)
}

func literalMediaURL(value string) bool {
	if len(value) > 2048 || !strings.HasPrefix(value, "https://") || !utf8.ValidString(value) {
		return false
	}

	u, err := url.Parse(value)

	return err == nil && u.Hostname() != "" && u.User == nil
}

func wireStrings(values []string) []any {
	wire := make([]any, len(values))
	for i, value := range values {
		wire[i] = value
	}

	return wire
}

func childList(id, kind string, children []string) BasicComponent {
	if len(children) == 0 || len(children) > maxComponents {
		return invalidBasicComponent()
	}

	children = slices.Clone(children)

	return basicNode(id, kind, true, children, func() map[string]any {
		return map[string]any{"children": wireStrings(children)}
	})
}

// NewText builds a Text node with a literal string.
func NewText(id, text string) BasicComponent {
	return basicNode(id, "Text", literalText(text), nil, func() map[string]any {
		return map[string]any{"text": text}
	})
}

// NewRow builds a Row with a nonempty ordered list of children.
func NewRow(id string, children ...string) BasicComponent { return childList(id, "Row", children) }

// NewColumn builds a Column with a nonempty ordered list of children.
func NewColumn(id string, children ...string) BasicComponent {
	return childList(id, "Column", children)
}

// NewList builds a List with a nonempty ordered list of children.
func NewList(id string, children ...string) BasicComponent { return childList(id, "List", children) }

// NewCard builds a Card containing one child.
func NewCard(id, child string) BasicComponent {
	return basicNode(id, "Card", true, []string{child}, func() map[string]any {
		return map[string]any{fieldChild: child}
	})
}

// Tab identifies a tab's title and child within one surface.
type Tab struct {
	Title string
	Child string
}

// NewTabs builds an ordered, nonempty set of tab panels.
func NewTabs(id string, tabs ...Tab) BasicComponent {
	if len(tabs) == 0 || len(tabs) > maxBasicGroupItems {
		return invalidBasicComponent()
	}

	tabs = slices.Clone(tabs)

	children := make([]string, 0, len(tabs))
	for _, tab := range tabs {
		if !literalLabel(tab.Title) {
			return invalidBasicComponent()
		}

		children = append(children, tab.Child)
	}

	return basicNode(id, "Tabs", true, children, func() map[string]any {
		wire := make([]any, 0, len(tabs))
		for _, tab := range tabs {
			wire = append(wire, map[string]any{"title": tab.Title, fieldChild: tab.Child})
		}

		return map[string]any{"tabs": wire}
	})
}

// NewModal builds a Modal with separate trigger and content children.
func NewModal(id, trigger, content string) BasicComponent {
	return basicNode(id, "Modal", true, []string{trigger, content}, func() map[string]any {
		return map[string]any{"trigger": trigger, "content": content}
	})
}

// NewDivider builds a horizontal Divider.
func NewDivider(id string) BasicComponent {
	return basicNode(id, "Divider", true, nil, func() map[string]any { return map[string]any{} })
}

// NewImage builds an Image with a credential-free HTTPS URL and description.
func NewImage(id, mediaURL, description string) BasicComponent {
	return basicNode(id, "Image", literalMediaURL(mediaURL) && literalLabel(description), nil, func() map[string]any {
		return map[string]any{fieldURL: mediaURL, "description": description}
	})
}

// NewIcon builds an Icon using a standard Basic Catalog name.
func NewIcon(id, name string) BasicComponent {
	return basicNode(id, "Icon", validIcon(name), nil, func() map[string]any {
		return map[string]any{"name": name}
	})
}

func validIcon(name string) bool {
	switch name {
	case "accountCircle", "add", "arrowBack", "arrowForward", "attachFile", "calendarToday", "call",
		"camera", "check", "close", "delete", "download", "edit", "event",
		"error", "fastForward", "favorite", "favoriteOff", "folder", "help", "home",
		"info", "locationOn", "lock", "lockOpen", "mail", "menu", "moreVert",
		"moreHoriz", "notificationsOff", "notifications", "pause", "payment", "person", "phone",
		"photo", "play", "print", "refresh", "rewind", "search", "send",
		"settings", "share", "shoppingCart", "skipNext", "skipPrevious", "star", "starHalf",
		"starOff", "stop", "upload", "visibility", "visibilityOff", "volumeDown", "volumeMute",
		"volumeOff", "volumeUp", "warning":
		return true
	default:
		return false
	}
}

// NewVideo builds a Video with a credential-free HTTPS URL.
func NewVideo(id, mediaURL string) BasicComponent {
	return basicNode(id, "Video", literalMediaURL(mediaURL), nil, func() map[string]any {
		return map[string]any{fieldURL: mediaURL}
	})
}

// NewAudioPlayer builds an AudioPlayer with a credential-free HTTPS URL.
func NewAudioPlayer(id, mediaURL, description string) BasicComponent {
	return basicNode(id, "AudioPlayer", literalMediaURL(mediaURL) && literalLabel(description), nil, func() map[string]any {
		return map[string]any{fieldURL: mediaURL, "description": description}
	})
}

// NewButton declares a literal caller-owned event name, required by the
// standard Basic Catalog. The SDK neither dispatches nor authorizes it.
func NewButton(id, child, eventName string) BasicComponent {
	return basicNode(id, "Button", validID(eventName), []string{child}, func() map[string]any {
		return map[string]any{fieldChild: child, "action": map[string]any{"event": map[string]any{"name": eventName}}}
	})
}

// NewTextField builds a TextField with a literal initial value.
func NewTextField(id, label, value string) BasicComponent {
	return basicNode(id, "TextField", literalLabel(label) && literalText(value), nil, func() map[string]any {
		return map[string]any{fieldLabel: label, fieldValue: value}
	})
}

// NewCheckBox builds a CheckBox with a literal initial value.
func NewCheckBox(id, label string, checked bool) BasicComponent {
	return basicNode(id, "CheckBox", literalLabel(label), nil, func() map[string]any {
		return map[string]any{fieldLabel: label, fieldValue: checked}
	})
}

// ChoiceOption is a displayed label and stable value for a ChoicePicker.
type ChoiceOption struct {
	Label string
	Value string
}

// NewChoicePicker builds a ChoicePicker with literal options and selections.
func NewChoicePicker(id, label string, options []ChoiceOption, selected []string, multiple bool) BasicComponent {
	if !literalLabel(label) || len(options) == 0 || len(options) > maxBasicGroupItems ||
		len(selected) > len(options) || (!multiple && len(selected) > 1) {
		return invalidBasicComponent()
	}

	values := make(map[string]bool, len(options))
	for _, option := range options {
		if !literalLabel(option.Label) || !validID(option.Value) {
			return invalidBasicComponent()
		}

		if _, exists := values[option.Value]; exists {
			return invalidBasicComponent()
		}

		values[option.Value] = true
	}

	for _, value := range selected {
		if !values[value] {
			return invalidBasicComponent()
		}

		values[value] = false
	}

	options = slices.Clone(options)
	selected = slices.Clone(selected)

	return basicNode(id, "ChoicePicker", true, nil, func() map[string]any {
		wireOptions := make([]any, 0, len(options))
		for _, option := range options {
			wireOptions = append(wireOptions, map[string]any{fieldLabel: option.Label, fieldValue: option.Value})
		}

		variant := "mutuallyExclusive"
		if multiple {
			variant = "multipleSelection"
		}

		return map[string]any{
			fieldLabel: label, "options": wireOptions, fieldValue: wireStrings(selected), "variant": variant,
		}
	})
}

// NewSlider builds a Slider with finite bounds and an initial value in range.
func NewSlider(id, label string, minimum, maximum, value float64) BasicComponent {
	valid := literalLabel(label) && !math.IsInf(minimum, 0) && !math.IsInf(maximum, 0) && maximum > minimum && value >= minimum && value <= maximum

	return basicNode(id, "Slider", valid, nil, func() map[string]any {
		return map[string]any{fieldLabel: label, "min": minimum, "max": maximum, fieldValue: value}
	})
}

// DateTimeMode specifies which fields are enabled in a DateTimeInput.
type DateTimeMode uint8

const (
	// DateOnly expects YYYY-MM-DD.
	DateOnly DateTimeMode = iota + 1
	// TimeOnly expects HH:mm or HH:mm:ss.
	TimeOnly
	// DateAndTime expects an RFC 3339 timestamp with a timezone.
	DateAndTime
)

// NewDateTimeInput builds a DateTimeInput with a literal initial value.
// An empty value leaves the input blank.
func NewDateTimeInput(id, label, value string, mode DateTimeMode) BasicComponent {
	if !literalLabel(label) || len(value) > 64 {
		return invalidBasicComponent()
	}
	var layout string

	switch mode {
	case DateOnly:
		layout = time.DateOnly
	case TimeOnly:
		layout = "15:04"
		if len(value) != len(layout) {
			layout = time.TimeOnly
		}
	case DateAndTime:
		layout = time.RFC3339
	default:
		return invalidBasicComponent()
	}

	if value != "" {
		if _, err := time.Parse(layout, value); err != nil {
			return invalidBasicComponent()
		}
	}

	return basicNode(id, "DateTimeInput", true, nil, func() map[string]any {
		return map[string]any{
			fieldLabel: label, fieldValue: value, "enableDate": mode != TimeOnly, "enableTime": mode != DateOnly,
		}
	})
}
