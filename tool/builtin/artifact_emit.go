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

	"github.com/rs/xid"

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

const artifactEmitDescription = `Emit an artifact containing text outputs or results of your work. Use this to provide structured text outputs to the user.

WHEN TO USE:
- When you need to provide completed text outputs (reports, summaries, analysis)
- When returning structured text results from computation
- When delivering final text deliverables to the user

FUNCTIONALITY:
- Create new artifacts with name/description and text content
- Append text to existing artifacts using append_to_artifact_id

EXAMPLES:
New artifact: {"name": "Analysis Report", "description": "Summary of findings", "text": "Analysis results...\n\nConclusions..."}
Append to existing: {"append_to_artifact_id": "artifact-123", "text": "Additional findings..."}`

// NewArtifactEmitTool returns the artifact_emit built-in.
func NewArtifactEmitTool() tool.Tool {
	return tool.Must(tool.Func(
		tool.Spec{
			Name:        "artifact_emit",
			Description: artifactEmitDescription,
			InputSchema: mustMarshal(artifactInputSchema),
		},
		func(_ context.Context, in EmitArtifactInput) (tool.Result[ArtifactEmitOutput], error) {
			if in.Name == "" {
				return tool.Result[ArtifactEmitOutput]{}, errors.New("artifact must have non-empty name")
			}

			if in.Description == "" {
				return tool.Result[ArtifactEmitOutput]{}, errors.New("artifact must have non-empty description")
			}

			if in.Text == "" {
				return tool.Result[ArtifactEmitOutput]{}, errors.New("artifact must have non-empty text content")
			}

			out := ArtifactEmitOutput{ArtifactID: "artifact-" + xid.New().String()}

			return tool.Done(out, tool.Action{
				Kind: tool.ActionArtifact,
				Artifact: &tool.ArtifactAction{
					ID:          out.ArtifactID,
					Name:        in.Name,
					Description: in.Description,
					MediaType:   "text/plain",
					Data:        []byte(in.Text),
				},
			}), nil
		},
	))
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

func mustMarshal(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		// Schema is a static literal — a marshal failure here means
		// the source itself is malformed and the program cannot start.
		panic(err) //nolint:forbidigo // init-time programmer error
	}

	return b
}
