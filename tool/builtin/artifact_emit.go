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

package builtin

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/rs/xid"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// EmitArtifactInput represents simplified artifact input for text-only artifacts.
type EmitArtifactInput struct {
	Name        string `json:"name"        jsonschema_description:"Name for the artifact"`
	Description string `json:"description" jsonschema_description:"Description of the artifact"`
	Text        string `json:"text"        jsonschema_description:"Text content for the artifact"`
}

// ArtifactEmitOutput represents the response from artifact emission.
type ArtifactEmitOutput struct {
	ArtifactID string `json:"artifact_id"`
}

// ArtifactEmitTool implements a tool for emitting A2A artifacts.
type ArtifactEmitTool struct{}

// Ensure ArtifactEmitTool implements tool.Tool interface.
var _ tool.Tool = (*ArtifactEmitTool)(nil)

// NewArtifactEmitTool creates a new ArtifactEmitTool instance.
func NewArtifactEmitTool() tool.Tool {
	return &ArtifactEmitTool{}
}

// Definition returns the tool definition for the LLM.
func (*ArtifactEmitTool) Definition() llm.ToolDefinition {
	// Convert schema to JSON
	schemaBytes, err := json.Marshal(artifactInputSchema)
	if err != nil {
		// Fallback to empty schema if marshaling fails
		schemaBytes = []byte("{}")
	}

	return llm.ToolDefinition{
		Name: "artifact_emit",
		Description: `Emit an artifact containing text outputs or results of your work. Use this to provide structured text outputs to the user.

WHEN TO USE:
- When you need to provide completed text outputs (reports, summaries, analysis)
- When returning structured text results from computation
- When delivering final text deliverables to the user

FUNCTIONALITY:
- Create new artifacts with name/description and text content
- Append text to existing artifacts using append_to_artifact_id

EXAMPLES:
New artifact: {"name": "Analysis Report", "description": "Summary of findings", "text": "Analysis results...\n\nConclusions..."}
Append to existing: {"append_to_artifact_id": "artifact-123", "text": "Additional findings..."}`,
		Parameters: schemaBytes,
		Type:       llm.ToolTypeFunction,
	}
}

// Execute processes the artifact emit request.
func (*ArtifactEmitTool) Execute(_ context.Context, args json.RawMessage) (json.RawMessage, error) {
	var input EmitArtifactInput

	err := json.Unmarshal(args, &input)
	if err != nil {
		return nil, fmt.Errorf("failed to parse artifact emit request: %w", err)
	}

	if input.Name == "" {
		return nil, errors.New("artifact must have non-empty name")
	}

	if input.Description == "" {
		return nil, errors.New("artifact must have non-empty description")
	}

	if input.Text == "" {
		return nil, errors.New("artifact must have non-empty text content")
	}

	// Create the response with properly typed artifact data
	output := ArtifactEmitOutput{
		ArtifactID: "artifact-" + xid.New().String(),
	}

	return json.Marshal(output)
}

// Manual JSON schema for EmitArtifactInput.
var artifactInputSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"name": map[string]any{
			"type":        "string",
			"description": "Name for the artifact",
		},
		"description": map[string]any{
			"type":        "string",
			"description": "Description of the artifact",
		},
		"text": map[string]any{
			"type":        "string",
			"description": "Text content for the artifact",
		},
	},
	"required":             []string{"name", "description", "text"},
	"additionalProperties": false,
}
