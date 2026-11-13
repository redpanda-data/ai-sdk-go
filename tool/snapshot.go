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

package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"slices"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Snapshot captures and combines registries into a read-only registry, preserving
// registration policies and executable identities. Nil inputs are ignored.
// Duplicate names and conflicting group metadata are rejected.
//
// Definition and configuration metadata are copied recursively through acyclic
// map[string]any, map[string]string, []any, []map[string]any, []string, []byte,
// and json.RawMessage values. Other values are opaque and must be immutable.
//
// Each registry resolves against ctx once. Its returned snapshot must keep its
// definitions, membership, and execution policies immutable.
// Executable tools are shared, not cloned; their own state must be concurrency-safe.
// Each input is captured independently, not atomically across registries.
func Snapshot(ctx context.Context, registries ...Registry) (Registry, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	combined := &registry{tools: make(map[string]*registeredTool), frozen: true}

	for i, source := range registries {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		if source == nil {
			continue
		}

		captured, err := source.Snapshot(ctx)
		if err != nil {
			return nil, fmt.Errorf("tool: snapshot input %d: %w", i, err)
		}

		if captured == nil {
			return nil, fmt.Errorf("tool: snapshot input %d returned nil", i)
		}

		if native, ok := captured.(*registry); ok {
			native.mu.RLock()
			err := combined.merge(native.tools)
			native.mu.RUnlock()

			if err != nil {
				return nil, err
			}

			continue
		}

		defs, err := captured.List(ctx)
		if err != nil {
			return nil, fmt.Errorf("tool: snapshot input %d: %w", i, err)
		}

		for _, def := range defs {
			t, err := captured.Get(ctx, def.Name)
			if err != nil {
				return nil, fmt.Errorf("tool: snapshot %q: %w", def.Name, err)
			}

			definition := cloneDefinition(def)
			entry := &registeredTool{
				tool: t, config: Config{Group: def.Group, Deferred: def.Deferred},
				definition: &definition, executor: captured,
			}

			if err := combined.addSnapshotEntry(def.Name, entry); err != nil {
				return nil, err
			}
		}
	}

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	return combined, nil
}

// Snapshot freezes this registry's membership, definitions, and registration
// policies. Subsequent registration changes do not affect the returned registry.
func (r *registry) Snapshot(ctx context.Context) (Registry, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if r.frozen {
		return r, nil
	}

	captured := &registry{tools: make(map[string]*registeredTool, len(r.tools)), config: r.config, frozen: true}
	for name, entry := range r.tools {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		copied := *entry
		definition := cloneDefinition(entry.toolDefinition())
		copied.definition = &definition
		copied.config.Metadata = cloneMetadata(entry.config.Metadata)
		captured.tools[name] = &copied
	}

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	return captured, nil
}

func (r *registry) merge(entries map[string]*registeredTool) error {
	for name, entry := range entries {
		if err := r.addSnapshotEntry(name, entry); err != nil {
			return err
		}
	}

	return nil
}

func (r *registry) addSnapshotEntry(name string, entry *registeredTool) error {
	if name == "" {
		return ErrToolNameEmpty
	}

	if _, exists := r.tools[name]; exists {
		return fmt.Errorf("%w: %q", ErrToolAlreadyRegistered, name)
	}

	if err := r.checkGroup(entry.config.Group); err != nil {
		return fmt.Errorf("%w for tool %q: %w", ErrInvalidToolConfig, name, err)
	}

	r.tools[name] = entry

	return nil
}

func (r *registeredTool) toolDefinition() llm.ToolDefinition {
	if r.definition != nil {
		return cloneDefinition(*r.definition)
	}

	return r.tool.Definition()
}

type snapshotTool struct {
	Tool

	definition llm.ToolDefinition
}

func (t *snapshotTool) Definition() llm.ToolDefinition {
	return cloneDefinition(t.definition)
}

func cloneDefinition(def llm.ToolDefinition) llm.ToolDefinition {
	def.Parameters = append([]byte(nil), def.Parameters...)
	def.Metadata = cloneMetadata(def.Metadata)

	return def
}

func cloneMetadata(metadata map[string]any) map[string]any {
	copied := maps.Clone(metadata)
	for key, value := range copied {
		copied[key] = cloneMetadataValue(value)
	}

	return copied
}

func cloneMetadataValue(value any) any {
	switch value := value.(type) {
	case map[string]any:
		return cloneMetadata(value)
	case map[string]string:
		return maps.Clone(value)
	case []any:
		copied := slices.Clone(value)
		for i, item := range copied {
			copied[i] = cloneMetadataValue(item)
		}

		return copied
	case []map[string]any:
		copied := slices.Clone(value)
		for i, item := range copied {
			copied[i] = cloneMetadata(item)
		}

		return copied
	case []string:
		return slices.Clone(value)
	case []byte:
		return slices.Clone(value)
	case json.RawMessage:
		return slices.Clone(value)
	default:
		return value
	}
}
