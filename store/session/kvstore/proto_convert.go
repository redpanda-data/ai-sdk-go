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
	"errors"
	"fmt"

	"google.golang.org/protobuf/types/known/structpb"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	llmpb "github.com/redpanda-data/ai-sdk-go/store/session/kvstore/proto/gen/go/redpanda/llm/v1"
)

// getToProtoConverter returns the conversion function from State to proto.
func getToProtoConverter() func(*session.State) (*llmpb.SessionState, error) {
	return toProtoSessionState
}

// getFromProtoConverter returns the conversion function from proto to State.
func getFromProtoConverter() func(*llmpb.SessionState) (*session.State, error) {
	return FromProtoSessionState
}

// toProtoSessionState converts a Go session.State to protobuf.
func toProtoSessionState(s *session.State) (*llmpb.SessionState, error) {
	if s == nil {
		return nil, errors.New("cannot convert nil State to proto")
	}

	// Convert messages
	pbMessages := make([]*llmpb.Message, len(s.Messages))
	for i, msg := range s.Messages {
		pbMsg, err := toProtoMessage(&msg)
		if err != nil {
			return nil, fmt.Errorf("convert message %d: %w", i, err)
		}

		pbMessages[i] = pbMsg
	}

	// Convert metadata
	var pbMetadata *structpb.Struct

	if s.Metadata != nil {
		var err error

		pbMetadata, err = structpb.NewStruct(s.Metadata)
		if err != nil {
			return nil, fmt.Errorf("convert metadata: %w", err)
		}
	}

	return &llmpb.SessionState{
		Id:             s.ID,
		ConversationId: s.ConversationID,
		Messages:       pbMessages,
		Metadata:       pbMetadata,
	}, nil
}

// FromProtoSessionState converts a protobuf SessionState to Go session.State.
// Exported for tests that need to parse protojson test fixtures.
func FromProtoSessionState(pb *llmpb.SessionState) (*session.State, error) {
	if pb == nil {
		return nil, errors.New("cannot convert nil proto SessionState")
	}

	// Convert messages
	messages := make([]llm.Message, len(pb.Messages))
	for i, pbMsg := range pb.Messages {
		msg, err := fromProtoMessage(pbMsg)
		if err != nil {
			return nil, fmt.Errorf("convert message %d: %w", i, err)
		}

		messages[i] = *msg
	}

	// Convert metadata
	var metadata map[string]any
	if pb.Metadata != nil {
		metadata = pb.Metadata.AsMap()
	}

	return &session.State{
		ID:             pb.Id,
		ConversationID: pb.ConversationId,
		Messages:       messages,
		Metadata:       metadata,
	}, nil
}

// toProtoMessage converts llm.Message to proto.
func toProtoMessage(msg *llm.Message) (*llmpb.Message, error) {
	if msg == nil {
		return &llmpb.Message{}, nil
	}

	// Convert role
	pbRole, err := toProtoRole(msg.Role)
	if err != nil {
		return nil, err
	}

	// Convert content parts
	pbParts := make([]*llmpb.Part, len(msg.Content))
	for i, part := range msg.Content {
		pbPart, err := toProtoPart(part)
		if err != nil {
			return nil, fmt.Errorf("convert part %d: %w", i, err)
		}

		pbParts[i] = pbPart
	}

	return &llmpb.Message{
		Role:    pbRole,
		Content: pbParts,
	}, nil
}

// fromProtoMessage converts proto Message to llm.Message.
func fromProtoMessage(pb *llmpb.Message) (*llm.Message, error) {
	if pb == nil {
		return &llm.Message{}, nil
	}

	// Convert role
	role, err := fromProtoRole(pb.Role)
	if err != nil {
		return nil, err
	}

	// Convert content parts
	parts := make([]llm.Part, len(pb.Content))
	for i, pbPart := range pb.Content {
		part, err := fromProtoPart(pbPart)
		if err != nil {
			return nil, fmt.Errorf("convert part %d: %w", i, err)
		}

		parts[i] = part
	}

	return &llm.Message{
		Role:    role,
		Content: parts,
	}, nil
}

// toProtoPart converts llm.Part to proto with oneof population.
func toProtoPart(p llm.Part) (*llmpb.Part, error) {
	if p == nil {
		return &llmpb.Part{}, nil
	}

	pbPart := &llmpb.Part{}

	switch v := p.(type) {
	case *llm.TextPart:
		if v == nil {
			return nil, errors.New("nil *TextPart")
		}

		pbPart.Kind = llmpb.PartKind_PART_KIND_TEXT
		pbPart.Data = &llmpb.Part_Text{Text: v.Text}

	case *llm.ToolRequestPart:
		if v == nil {
			return nil, errors.New("nil *ToolRequestPart")
		}

		pbPart.Kind = llmpb.PartKind_PART_KIND_TOOL_REQUEST
		pbPart.Data = &llmpb.Part_ToolRequest{
			ToolRequest: &llmpb.ToolRequest{
				Id:        v.ID,
				Name:      v.Name,
				Arguments: []byte(v.Arguments),
			},
		}

		if v.Metadata != nil {
			meta, err := structpb.NewStruct(v.Metadata)
			if err != nil {
				return nil, fmt.Errorf("convert tool request metadata: %w", err)
			}

			pbPart.Metadata = meta
		}

	case *llm.ToolResponsePart:
		if v == nil {
			return nil, errors.New("nil *ToolResponsePart")
		}

		pbPart.Kind = llmpb.PartKind_PART_KIND_TOOL_RESPONSE

		errStr := ""
		if v.IsError {
			errStr = string(v.Result)
		}

		pbPart.Data = &llmpb.Part_ToolResponse{
			ToolResponse: &llmpb.ToolResponse{
				Id:     v.ID,
				Name:   v.Name,
				Result: []byte(v.Result),
				Error:  errStr,
			},
		}

	case *llm.ReasoningPart:
		if v == nil {
			return nil, errors.New("nil *ReasoningPart")
		}

		pbPart.Kind = llmpb.PartKind_PART_KIND_REASONING

		var pbReasoningMeta *structpb.Struct

		if v.Metadata != nil {
			var err error

			pbReasoningMeta, err = structpb.NewStruct(v.Metadata)
			if err != nil {
				return nil, fmt.Errorf("convert reasoning metadata: %w", err)
			}
		}

		pbPart.Data = &llmpb.Part_ReasoningTrace{
			ReasoningTrace: &llmpb.ReasoningTrace{
				Id:       v.Signature,
				Text:     v.Text,
				Metadata: pbReasoningMeta,
			},
		}

	default:
		return nil, fmt.Errorf("unknown Part type: %T", p)
	}

	return pbPart, nil
}

// fromProtoPart converts proto Part to llm.Part with oneof extraction.
func fromProtoPart(pb *llmpb.Part) (llm.Part, error) {
	if pb == nil {
		return nil, errors.New("nil proto Part")
	}

	// Extract data from oneof
	switch data := pb.Data.(type) {
	case *llmpb.Part_Text:
		return &llm.TextPart{Text: data.Text}, nil

	case *llmpb.Part_ToolRequest:
		if data.ToolRequest == nil {
			return nil, errors.New("Part_ToolRequest has nil ToolRequest")
		}

		out := &llm.ToolRequestPart{
			ID:        data.ToolRequest.Id,
			Name:      data.ToolRequest.Name,
			Arguments: data.ToolRequest.Arguments,
		}

		if pb.Metadata != nil {
			out.Metadata = pb.Metadata.AsMap()
		}

		return out, nil

	case *llmpb.Part_ToolResponse:
		if data.ToolResponse == nil {
			return nil, errors.New("Part_ToolResponse has nil ToolResponse")
		}

		out := &llm.ToolResponsePart{
			ID:     data.ToolResponse.Id,
			Name:   data.ToolResponse.Name,
			Result: data.ToolResponse.Result,
		}

		// Older payloads stored error text in Error and result in Result; rehydrate as IsError.
		if data.ToolResponse.Error != "" {
			out.IsError = true
			if len(out.Result) == 0 {
				out.Result = []byte(data.ToolResponse.Error)
			}
		}

		return out, nil

	case *llmpb.Part_ReasoningTrace:
		if data.ReasoningTrace == nil {
			return nil, errors.New("Part_ReasoningTrace has nil ReasoningTrace")
		}

		out := &llm.ReasoningPart{
			Signature: data.ReasoningTrace.Id,
			Text:      data.ReasoningTrace.Text,
		}

		if data.ReasoningTrace.Metadata != nil {
			out.Metadata = data.ReasoningTrace.Metadata.AsMap()
		}

		return out, nil

	case nil:
		return nil, errors.New("part has no data set")

	default:
		return nil, fmt.Errorf("unknown Part data type: %T", data)
	}
}

// toProtoRole converts llm.MessageRole to proto enum.
func toProtoRole(role llm.MessageRole) (llmpb.MessageRole, error) {
	switch role {
	case llm.RoleUser:
		return llmpb.MessageRole_MESSAGE_ROLE_USER, nil
	case llm.RoleAssistant:
		return llmpb.MessageRole_MESSAGE_ROLE_ASSISTANT, nil
	case llm.RoleSystem:
		return llmpb.MessageRole_MESSAGE_ROLE_SYSTEM, nil
	default:
		return llmpb.MessageRole_MESSAGE_ROLE_UNSPECIFIED, fmt.Errorf("unknown MessageRole: %v", role)
	}
}

// fromProtoRole converts proto MessageRole to llm.MessageRole.
func fromProtoRole(pbRole llmpb.MessageRole) (llm.MessageRole, error) {
	switch pbRole {
	case llmpb.MessageRole_MESSAGE_ROLE_USER:
		return llm.RoleUser, nil
	case llmpb.MessageRole_MESSAGE_ROLE_ASSISTANT:
		return llm.RoleAssistant, nil
	case llmpb.MessageRole_MESSAGE_ROLE_SYSTEM:
		return llm.RoleSystem, nil
	case llmpb.MessageRole_MESSAGE_ROLE_UNSPECIFIED:
		return "", errors.New("unspecified MessageRole")
	default:
		return "", fmt.Errorf("unknown MessageRole: %v", pbRole)
	}
}
