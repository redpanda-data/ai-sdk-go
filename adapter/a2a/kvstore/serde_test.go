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

package kvstore

import (
	"testing"
	"time"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2apb"
	"github.com/a2aproject/a2a-go/a2apb/pbconv"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/types/known/structpb"
)

func TestToProtoTask_Metadata(t *testing.T) {
	t.Parallel()

	now := time.Now().Truncate(time.Microsecond) // Truncate for proto timestamp precision

	task := &a2a.Task{
		ID:        "task-123",
		ContextID: "ctx-456",
		Status: a2a.TaskStatus{
			State:     a2a.TaskStateWorking,
			Timestamp: &now,
		},
		Metadata: map[string]any{
			"string_val":  "hello",
			"int_val":     float64(42), // JSON numbers are float64
			"float_val":   3.14,
			"bool_val":    true,
			"null_val":    nil,
			"nested_map":  map[string]any{"inner_key": "inner_value"},
			"string_list": []any{"a", "b", "c"},
		},
	}

	pb, err := pbconv.ToProtoTask(task)
	require.NoError(t, err)

	// Verify basic fields
	assert.Equal(t, "task-123", pb.Id)
	assert.Equal(t, "ctx-456", pb.ContextId)
	assert.Equal(t, a2apb.TaskState_TASK_STATE_WORKING, pb.Status.State)

	// Verify metadata is converted to structpb
	require.NotNil(t, pb.Metadata)
	require.NotNil(t, pb.Metadata.Fields)

	// Check string value
	strVal := pb.Metadata.Fields["string_val"]
	require.NotNil(t, strVal)
	assert.Equal(t, "hello", strVal.GetStringValue())

	// Check number value
	intVal := pb.Metadata.Fields["int_val"]
	require.NotNil(t, intVal)
	assert.InDelta(t, float64(42), intVal.GetNumberValue(), 0)

	// Check float value
	floatVal := pb.Metadata.Fields["float_val"]
	require.NotNil(t, floatVal)
	assert.InDelta(t, 3.14, floatVal.GetNumberValue(), 0.001)

	// Check bool value
	boolVal := pb.Metadata.Fields["bool_val"]
	require.NotNil(t, boolVal)
	assert.True(t, boolVal.GetBoolValue())

	// Check null value
	nullVal := pb.Metadata.Fields["null_val"]
	require.NotNil(t, nullVal)
	assert.Equal(t, structpb.NullValue_NULL_VALUE, nullVal.GetNullValue())

	// Check nested map
	nestedMap := pb.Metadata.Fields["nested_map"]
	require.NotNil(t, nestedMap)
	require.NotNil(t, nestedMap.GetStructValue())
	innerKey := nestedMap.GetStructValue().Fields["inner_key"]
	require.NotNil(t, innerKey)
	assert.Equal(t, "inner_value", innerKey.GetStringValue())

	// Check list
	listVal := pb.Metadata.Fields["string_list"]
	require.NotNil(t, listVal)
	require.NotNil(t, listVal.GetListValue())
	assert.Len(t, listVal.GetListValue().Values, 3)
	assert.Equal(t, "a", listVal.GetListValue().Values[0].GetStringValue())
	assert.Equal(t, "b", listVal.GetListValue().Values[1].GetStringValue())
	assert.Equal(t, "c", listVal.GetListValue().Values[2].GetStringValue())
}

func TestToProtoTask_MessageMetadata(t *testing.T) {
	t.Parallel()

	task := &a2a.Task{
		ID:        "task-msg-meta",
		ContextID: "ctx-1",
		Status: a2a.TaskStatus{
			State: a2a.TaskStateCompleted,
		},
		History: []*a2a.Message{
			{
				Role: a2a.MessageRoleUser,
				Parts: []a2a.Part{
					a2a.TextPart{Text: "Hello"},
				},
				Metadata: map[string]any{
					"user_id":    "user-123",
					"request_id": "req-456",
				},
			},
		},
	}

	pb, err := pbconv.ToProtoTask(task)
	require.NoError(t, err)

	require.Len(t, pb.History, 1)
	msg := pb.History[0]

	require.NotNil(t, msg.Metadata)
	assert.Equal(t, "user-123", msg.Metadata.Fields["user_id"].GetStringValue())
	assert.Equal(t, "req-456", msg.Metadata.Fields["request_id"].GetStringValue())
}

func TestToProtoTask_ArtifactMetadata(t *testing.T) {
	t.Parallel()

	task := &a2a.Task{
		ID:        "task-artifact-meta",
		ContextID: "ctx-1",
		Status: a2a.TaskStatus{
			State: a2a.TaskStateCompleted,
		},
		Artifacts: []*a2a.Artifact{
			{
				ID:          "artifact-1",
				Name:        "result.json",
				Description: "Test artifact",
				Parts: []a2a.Part{
					a2a.TextPart{Text: `{"result": "success"}`},
				},
				Metadata: map[string]any{
					"mime_type": "application/json",
					"size":      float64(1024),
				},
			},
		},
	}

	pb, err := pbconv.ToProtoTask(task)
	require.NoError(t, err)

	require.Len(t, pb.Artifacts, 1)
	artifact := pb.Artifacts[0]

	assert.Equal(t, "artifact-1", artifact.ArtifactId)
	assert.Equal(t, "result.json", artifact.Name)
	assert.Equal(t, "Test artifact", artifact.Description)

	require.NotNil(t, artifact.Metadata)
	assert.Equal(t, "application/json", artifact.Metadata.Fields["mime_type"].GetStringValue())
	assert.InDelta(t, float64(1024), artifact.Metadata.Fields["size"].GetNumberValue(), 0)
}

func TestFromProtoTask_Metadata(t *testing.T) {
	t.Parallel()

	metadata, err := structpb.NewStruct(map[string]any{
		"key1": "value1",
		"key2": float64(123),
		"nested": map[string]any{
			"inner": "data",
		},
	})
	require.NoError(t, err)

	pb := &a2apb.Task{
		Id:        "task-from-proto",
		ContextId: "ctx-from-proto",
		Status: &a2apb.TaskStatus{
			State: a2apb.TaskState_TASK_STATE_COMPLETED,
		},
		Metadata: metadata,
	}

	task, err := pbconv.FromProtoTask(pb)
	require.NoError(t, err)

	assert.Equal(t, a2a.TaskID("task-from-proto"), task.ID)
	assert.Equal(t, "ctx-from-proto", task.ContextID)
	assert.Equal(t, a2a.TaskStateCompleted, task.Status.State)

	require.NotNil(t, task.Metadata)
	assert.Equal(t, "value1", task.Metadata["key1"])
	assert.InDelta(t, float64(123), task.Metadata["key2"], 0)

	nested, ok := task.Metadata["nested"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "data", nested["inner"])
}

// TestMetadata_NilVsEmpty documents the behavior of nil vs empty metadata at each level.
// This is critical for understanding serialization semantics.
func TestMetadata_NilVsEmpty(t *testing.T) {
	t.Parallel()

	t.Run("task_metadata_nil", func(t *testing.T) {
		t.Parallel()

		task := &a2a.Task{
			ID:        "task-nil-meta",
			ContextID: "ctx-1",
			Status:    a2a.TaskStatus{State: a2a.TaskStateCompleted},
			Metadata:  nil, // explicitly nil
		}

		pb, err := pbconv.ToProtoTask(task)
		require.NoError(t, err)

		// nil metadata -> nil proto Metadata field
		assert.Nil(t, pb.Metadata, "nil Go metadata should produce nil proto Metadata")

		// Round-trip back
		result, err := pbconv.FromProtoTask(pb)
		require.NoError(t, err)
		assert.Nil(t, result.Metadata, "nil proto Metadata should produce nil Go metadata")
	})

	t.Run("task_metadata_empty", func(t *testing.T) {
		t.Parallel()

		task := &a2a.Task{
			ID:        "task-empty-meta",
			ContextID: "ctx-1",
			Status:    a2a.TaskStatus{State: a2a.TaskStateCompleted},
			Metadata:  map[string]any{}, // empty map
		}

		pb, err := pbconv.ToProtoTask(task)
		require.NoError(t, err)

		// empty map -> empty Struct (not nil)
		require.NotNil(t, pb.Metadata, "empty Go metadata should produce non-nil proto Metadata")
		assert.Empty(t, pb.Metadata.Fields, "empty Go metadata should produce empty Fields")

		// Round-trip back
		result, err := pbconv.FromProtoTask(pb)
		require.NoError(t, err)
		// Note: empty Struct may come back as empty map or nil depending on implementation
		// Document actual behavior:
		if result.Metadata == nil {
			t.Log("BEHAVIOR: empty proto Struct -> nil Go metadata")
		} else {
			t.Log("BEHAVIOR: empty proto Struct -> empty Go metadata map")
			assert.Empty(t, result.Metadata)
		}
	})

	t.Run("message_metadata_nil", func(t *testing.T) {
		t.Parallel()

		task := &a2a.Task{
			ID:        "task-msg-nil-meta",
			ContextID: "ctx-1",
			Status:    a2a.TaskStatus{State: a2a.TaskStateCompleted},
			History: []*a2a.Message{
				{
					Role:     a2a.MessageRoleUser,
					Parts:    []a2a.Part{a2a.TextPart{Text: "Hello"}},
					Metadata: nil,
				},
			},
		}

		pb, err := pbconv.ToProtoTask(task)
		require.NoError(t, err)

		require.Len(t, pb.History, 1)
		assert.Nil(t, pb.History[0].Metadata, "nil message metadata should produce nil proto Metadata")

		result, err := pbconv.FromProtoTask(pb)
		require.NoError(t, err)
		require.Len(t, result.History, 1)
		assert.Nil(t, result.History[0].Metadata, "nil proto message Metadata should produce nil Go metadata")
	})

	t.Run("message_metadata_empty", func(t *testing.T) {
		t.Parallel()

		task := &a2a.Task{
			ID:        "task-msg-empty-meta",
			ContextID: "ctx-1",
			Status:    a2a.TaskStatus{State: a2a.TaskStateCompleted},
			History: []*a2a.Message{
				{
					Role:     a2a.MessageRoleUser,
					Parts:    []a2a.Part{a2a.TextPart{Text: "Hello"}},
					Metadata: map[string]any{},
				},
			},
		}

		pb, err := pbconv.ToProtoTask(task)
		require.NoError(t, err)

		require.Len(t, pb.History, 1)
		require.NotNil(t, pb.History[0].Metadata)
		assert.Empty(t, pb.History[0].Metadata.Fields)

		result, err := pbconv.FromProtoTask(pb)
		require.NoError(t, err)
		require.Len(t, result.History, 1)

		if result.History[0].Metadata == nil {
			t.Log("BEHAVIOR: empty proto message Struct -> nil Go metadata")
		} else {
			t.Log("BEHAVIOR: empty proto message Struct -> empty Go metadata map")
		}
	})

	t.Run("artifact_metadata_nil", func(t *testing.T) {
		t.Parallel()

		task := &a2a.Task{
			ID:        "task-art-nil-meta",
			ContextID: "ctx-1",
			Status:    a2a.TaskStatus{State: a2a.TaskStateCompleted},
			Artifacts: []*a2a.Artifact{
				{
					ID:       "art-1",
					Name:     "test.txt",
					Parts:    []a2a.Part{a2a.TextPart{Text: "content"}},
					Metadata: nil,
				},
			},
		}

		pb, err := pbconv.ToProtoTask(task)
		require.NoError(t, err)

		require.Len(t, pb.Artifacts, 1)
		assert.Nil(t, pb.Artifacts[0].Metadata, "nil artifact metadata should produce nil proto Metadata")

		result, err := pbconv.FromProtoTask(pb)
		require.NoError(t, err)
		require.Len(t, result.Artifacts, 1)
		assert.Nil(t, result.Artifacts[0].Metadata)
	})

	t.Run("artifact_metadata_empty", func(t *testing.T) {
		t.Parallel()

		task := &a2a.Task{
			ID:        "task-art-empty-meta",
			ContextID: "ctx-1",
			Status:    a2a.TaskStatus{State: a2a.TaskStateCompleted},
			Artifacts: []*a2a.Artifact{
				{
					ID:       "art-1",
					Name:     "test.txt",
					Parts:    []a2a.Part{a2a.TextPart{Text: "content"}},
					Metadata: map[string]any{},
				},
			},
		}

		pb, err := pbconv.ToProtoTask(task)
		require.NoError(t, err)

		require.Len(t, pb.Artifacts, 1)
		require.NotNil(t, pb.Artifacts[0].Metadata)
		assert.Empty(t, pb.Artifacts[0].Metadata.Fields)

		result, err := pbconv.FromProtoTask(pb)
		require.NoError(t, err)
		require.Len(t, result.Artifacts, 1)

		if result.Artifacts[0].Metadata == nil {
			t.Log("BEHAVIOR: empty proto artifact Struct -> nil Go metadata")
		} else {
			t.Log("BEHAVIOR: empty proto artifact Struct -> empty Go metadata map")
		}
	})

	t.Run("part_metadata_nil", func(t *testing.T) {
		t.Parallel()

		task := &a2a.Task{
			ID:        "task-part-nil-meta",
			ContextID: "ctx-1",
			Status:    a2a.TaskStatus{State: a2a.TaskStateCompleted},
			History: []*a2a.Message{
				{
					Role: a2a.MessageRoleUser,
					Parts: []a2a.Part{
						a2a.TextPart{Text: "Hello", Metadata: nil},
					},
				},
			},
		}

		pb, err := pbconv.ToProtoTask(task)
		require.NoError(t, err)

		require.Len(t, pb.History, 1)
		require.Len(t, pb.History[0].Parts, 1)
		assert.Nil(t, pb.History[0].Parts[0].Metadata, "nil part metadata should produce nil proto Metadata")
	})

	t.Run("part_metadata_with_values", func(t *testing.T) {
		t.Parallel()

		task := &a2a.Task{
			ID:        "task-part-meta",
			ContextID: "ctx-1",
			Status:    a2a.TaskStatus{State: a2a.TaskStateCompleted},
			History: []*a2a.Message{
				{
					Role: a2a.MessageRoleUser,
					Parts: []a2a.Part{
						a2a.TextPart{
							Text: "Hello",
							Metadata: map[string]any{
								"language": "en",
								"tokens":   float64(5),
							},
						},
					},
				},
			},
		}

		pb, err := pbconv.ToProtoTask(task)
		require.NoError(t, err)

		require.Len(t, pb.History, 1)
		require.Len(t, pb.History[0].Parts, 1)
		require.NotNil(t, pb.History[0].Parts[0].Metadata)
		assert.Equal(t, "en", pb.History[0].Parts[0].Metadata.Fields["language"].GetStringValue())
		assert.InDelta(t, float64(5), pb.History[0].Parts[0].Metadata.Fields["tokens"].GetNumberValue(), 0)

		result, err := pbconv.FromProtoTask(pb)
		require.NoError(t, err)
		require.Len(t, result.History, 1)
		require.Len(t, result.History[0].Parts, 1)

		textPart, ok := result.History[0].Parts[0].(a2a.TextPart)
		require.True(t, ok)
		assert.Equal(t, "en", textPart.Metadata["language"])
		assert.InDelta(t, float64(5), textPart.Metadata["tokens"], 0)
	})

	t.Run("status_message_metadata", func(t *testing.T) {
		t.Parallel()

		task := &a2a.Task{
			ID:        "task-status-msg-meta",
			ContextID: "ctx-1",
			Status: a2a.TaskStatus{
				State: a2a.TaskStateWorking,
				Message: &a2a.Message{
					Role:  a2a.MessageRoleAgent,
					Parts: []a2a.Part{a2a.TextPart{Text: "Working..."}},
					Metadata: map[string]any{
						"progress": float64(50),
					},
				},
			},
		}

		pb, err := pbconv.ToProtoTask(task)
		require.NoError(t, err)

		require.NotNil(t, pb.Status.Update, "status message should be in Update field")
		require.NotNil(t, pb.Status.Update.Metadata)
		assert.InDelta(t, float64(50), pb.Status.Update.Metadata.Fields["progress"].GetNumberValue(), 0)

		result, err := pbconv.FromProtoTask(pb)
		require.NoError(t, err)
		require.NotNil(t, result.Status.Message)
		assert.InDelta(t, float64(50), result.Status.Message.Metadata["progress"], 0)
	})
}

// TestEmptyCollections documents behavior of empty vs nil slices.
func TestEmptyCollections(t *testing.T) {
	t.Parallel()

	t.Run("nil_history", func(t *testing.T) {
		t.Parallel()

		task := &a2a.Task{
			ID:        "task-nil-history",
			ContextID: "ctx-1",
			Status:    a2a.TaskStatus{State: a2a.TaskStateCompleted},
			History:   nil,
		}

		pb, err := pbconv.ToProtoTask(task)
		require.NoError(t, err)
		// BEHAVIOR: nil Go slice -> empty proto slice (not nil)
		// This is pbconv implementation detail
		assert.NotNil(t, pb.History, "BEHAVIOR: nil Go History -> empty proto History (not nil)")
		assert.Empty(t, pb.History)

		result, err := pbconv.FromProtoTask(pb)
		require.NoError(t, err)
		// BEHAVIOR: empty proto slice -> empty Go slice (not nil)
		assert.NotNil(t, result.History, "BEHAVIOR: empty proto History -> empty Go History (not nil)")
		assert.Empty(t, result.History)
		t.Log("BEHAVIOR: nil History is NOT preserved through round-trip (becomes empty slice)")
	})

	t.Run("empty_history", func(t *testing.T) {
		t.Parallel()

		task := &a2a.Task{
			ID:        "task-empty-history",
			ContextID: "ctx-1",
			Status:    a2a.TaskStatus{State: a2a.TaskStateCompleted},
			History:   []*a2a.Message{},
		}

		pb, err := pbconv.ToProtoTask(task)
		require.NoError(t, err)
		// Empty slice may become nil in proto
		if pb.History == nil {
			t.Log("BEHAVIOR: empty Go History -> nil proto History")
		} else {
			t.Log("BEHAVIOR: empty Go History -> empty proto History")
			assert.Empty(t, pb.History)
		}
	})

	t.Run("nil_artifacts", func(t *testing.T) {
		t.Parallel()

		task := &a2a.Task{
			ID:        "task-nil-artifacts",
			ContextID: "ctx-1",
			Status:    a2a.TaskStatus{State: a2a.TaskStateCompleted},
			Artifacts: nil,
		}

		pb, err := pbconv.ToProtoTask(task)
		require.NoError(t, err)
		// BEHAVIOR: nil Go slice -> empty proto slice (not nil)
		assert.NotNil(t, pb.Artifacts, "BEHAVIOR: nil Go Artifacts -> empty proto Artifacts (not nil)")
		assert.Empty(t, pb.Artifacts)

		result, err := pbconv.FromProtoTask(pb)
		require.NoError(t, err)
		// BEHAVIOR: empty proto slice -> empty Go slice (not nil)
		assert.NotNil(t, result.Artifacts, "BEHAVIOR: empty proto Artifacts -> empty Go Artifacts (not nil)")
		assert.Empty(t, result.Artifacts)
		t.Log("BEHAVIOR: nil Artifacts is NOT preserved through round-trip (becomes empty slice)")
	})

	t.Run("empty_artifacts", func(t *testing.T) {
		t.Parallel()

		task := &a2a.Task{
			ID:        "task-empty-artifacts",
			ContextID: "ctx-1",
			Status:    a2a.TaskStatus{State: a2a.TaskStateCompleted},
			Artifacts: []*a2a.Artifact{},
		}

		pb, err := pbconv.ToProtoTask(task)
		require.NoError(t, err)

		if pb.Artifacts == nil {
			t.Log("BEHAVIOR: empty Go Artifacts -> nil proto Artifacts")
		} else {
			t.Log("BEHAVIOR: empty Go Artifacts -> empty proto Artifacts")
		}
	})

	t.Run("message_nil_parts", func(t *testing.T) {
		t.Parallel()

		task := &a2a.Task{
			ID:        "task-nil-parts",
			ContextID: "ctx-1",
			Status:    a2a.TaskStatus{State: a2a.TaskStateCompleted},
			History: []*a2a.Message{
				{
					Role:  a2a.MessageRoleUser,
					Parts: nil,
				},
			},
		}

		pb, err := pbconv.ToProtoTask(task)
		require.NoError(t, err)
		require.Len(t, pb.History, 1)
		// BEHAVIOR: nil Go Parts -> empty proto Parts (not nil)
		assert.NotNil(t, pb.History[0].Parts, "BEHAVIOR: nil Go Parts -> empty proto Parts (not nil)")
		assert.Empty(t, pb.History[0].Parts)
		t.Log("BEHAVIOR: nil Parts is NOT preserved through round-trip (becomes empty slice)")
	})

	t.Run("message_empty_parts", func(t *testing.T) {
		t.Parallel()

		task := &a2a.Task{
			ID:        "task-empty-parts",
			ContextID: "ctx-1",
			Status:    a2a.TaskStatus{State: a2a.TaskStateCompleted},
			History: []*a2a.Message{
				{
					Role:  a2a.MessageRoleUser,
					Parts: []a2a.Part{},
				},
			},
		}

		pb, err := pbconv.ToProtoTask(task)
		require.NoError(t, err)
		require.Len(t, pb.History, 1)

		if pb.History[0].Parts == nil {
			t.Log("BEHAVIOR: empty Go Parts -> nil proto Parts")
		} else {
			t.Log("BEHAVIOR: empty Go Parts -> empty proto Parts")
		}
	})
}

func TestProtoRoundTrip_FullTask(t *testing.T) {
	t.Parallel()

	now := time.Now().Truncate(time.Microsecond)

	original := &a2a.Task{
		ID:        "task-roundtrip",
		ContextID: "ctx-roundtrip",
		Status: a2a.TaskStatus{
			State:     a2a.TaskStateWorking,
			Timestamp: &now,
			Message: &a2a.Message{
				Role:  a2a.MessageRoleAgent,
				Parts: []a2a.Part{a2a.TextPart{Text: "Processing..."}},
			},
		},
		History: []*a2a.Message{
			{
				Role:  a2a.MessageRoleUser,
				Parts: []a2a.Part{a2a.TextPart{Text: "Do something"}},
				Metadata: map[string]any{
					"source": "test",
				},
			},
		},
		Artifacts: []*a2a.Artifact{
			{
				ID:   "art-1",
				Name: "output.txt",
				Parts: []a2a.Part{
					a2a.TextPart{Text: "Output content"},
				},
				Metadata: map[string]any{
					"generated": true,
				},
			},
		},
		Metadata: map[string]any{
			"task_type":  "test",
			"priority":   float64(1),
			"tags":       []any{"unit-test", "roundtrip"},
			"config":     map[string]any{"timeout": float64(30)},
			"nullable":   nil,
			"is_enabled": true,
		},
	}

	// Convert to proto
	pb, err := pbconv.ToProtoTask(original)
	require.NoError(t, err)

	// Convert back
	result, err := pbconv.FromProtoTask(pb)
	require.NoError(t, err)

	// Verify all fields
	assert.Equal(t, original.ID, result.ID)
	assert.Equal(t, original.ContextID, result.ContextID)
	assert.Equal(t, original.Status.State, result.Status.State)

	// Verify history
	require.Len(t, result.History, 1)
	assert.Equal(t, a2a.MessageRoleUser, result.History[0].Role)
	assert.Equal(t, "test", result.History[0].Metadata["source"])

	// Verify artifacts
	require.Len(t, result.Artifacts, 1)
	assert.Equal(t, a2a.ArtifactID("art-1"), result.Artifacts[0].ID)
	assert.Equal(t, true, result.Artifacts[0].Metadata["generated"])

	// Verify task metadata
	assert.Equal(t, "test", result.Metadata["task_type"])
	assert.InDelta(t, float64(1), result.Metadata["priority"], 0)
	assert.Equal(t, true, result.Metadata["is_enabled"])
	assert.Nil(t, result.Metadata["nullable"])

	tags, ok := result.Metadata["tags"].([]any)
	require.True(t, ok)
	assert.Equal(t, []any{"unit-test", "roundtrip"}, tags)

	config, ok := result.Metadata["config"].(map[string]any)
	require.True(t, ok)
	assert.InDelta(t, float64(30), config["timeout"], 0)
}
