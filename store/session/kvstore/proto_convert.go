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
		Id:       s.ID,
		Messages: pbMessages,
		Metadata: pbMetadata,
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
		ID:       pb.Id,
		Messages: messages,
		Metadata: metadata,
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
	parts := make([]*llm.Part, len(pb.Content))
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
func toProtoPart(p *llm.Part) (*llmpb.Part, error) {
	if p == nil {
		return &llmpb.Part{}, nil
	}

	pbKind, err := toProtoPartKind(p.Kind)
	if err != nil {
		return nil, err
	}

	pbPart := &llmpb.Part{Kind: pbKind}

	// Populate oneof based on Kind
	switch p.Kind {
	case llm.PartText:
		pbPart.Data = &llmpb.Part_Text{Text: p.Text}

	case llm.PartToolRequest:
		if p.ToolRequest == nil {
			return nil, errors.New("PartToolRequest has nil ToolRequest")
		}

		pbPart.Data = &llmpb.Part_ToolRequest{
			ToolRequest: &llmpb.ToolRequest{
				Id:        p.ToolRequest.ID,
				Name:      p.ToolRequest.Name,
				Arguments: []byte(p.ToolRequest.Arguments),
			},
		}

	case llm.PartToolResponse:
		if p.ToolResponse == nil {
			return nil, errors.New("PartToolResponse has nil ToolResponse")
		}

		pbPart.Data = &llmpb.Part_ToolResponse{
			ToolResponse: &llmpb.ToolResponse{
				Id:     p.ToolResponse.ID,
				Name:   p.ToolResponse.Name,
				Result: []byte(p.ToolResponse.Result),
				Error:  p.ToolResponse.Error,
			},
		}

	case llm.PartReasoning:
		if p.ReasoningTrace == nil {
			return nil, errors.New("PartReasoning has nil ReasoningTrace")
		}

		var pbReasoningMeta *structpb.Struct

		if p.ReasoningTrace.Metadata != nil {
			var err error

			pbReasoningMeta, err = structpb.NewStruct(p.ReasoningTrace.Metadata)
			if err != nil {
				return nil, fmt.Errorf("convert reasoning metadata: %w", err)
			}
		}

		pbPart.Data = &llmpb.Part_ReasoningTrace{
			ReasoningTrace: &llmpb.ReasoningTrace{
				Id:       p.ReasoningTrace.ID,
				Text:     p.ReasoningTrace.Text,
				Metadata: pbReasoningMeta,
			},
		}

	default:
		return nil, fmt.Errorf("unknown PartKind: %v (expected text, tool_request, tool_response, or reasoning)", p.Kind)
	}

	// Convert part metadata
	if p.Metadata != nil {
		var err error

		pbPart.Metadata, err = structpb.NewStruct(p.Metadata)
		if err != nil {
			return nil, fmt.Errorf("convert part metadata: %w", err)
		}
	}

	return pbPart, nil
}

// fromProtoPart converts proto Part to llm.Part with oneof extraction.
func fromProtoPart(pb *llmpb.Part) (*llm.Part, error) {
	if pb == nil {
		return &llm.Part{}, nil
	}

	kind, err := fromProtoPartKind(pb.Kind)
	if err != nil {
		return nil, err
	}

	part := &llm.Part{Kind: kind}

	// Extract data from oneof
	switch data := pb.Data.(type) {
	case *llmpb.Part_Text:
		part.Text = data.Text

	case *llmpb.Part_ToolRequest:
		if data.ToolRequest == nil {
			return nil, errors.New("Part_ToolRequest has nil ToolRequest")
		}

		part.ToolRequest = &llm.ToolRequest{
			ID:        data.ToolRequest.Id,
			Name:      data.ToolRequest.Name,
			Arguments: data.ToolRequest.Arguments,
		}

	case *llmpb.Part_ToolResponse:
		if data.ToolResponse == nil {
			return nil, errors.New("Part_ToolResponse has nil ToolResponse")
		}

		part.ToolResponse = &llm.ToolResponse{
			ID:     data.ToolResponse.Id,
			Name:   data.ToolResponse.Name,
			Result: data.ToolResponse.Result,
			Error:  data.ToolResponse.Error,
		}

	case *llmpb.Part_ReasoningTrace:
		if data.ReasoningTrace == nil {
			return nil, errors.New("Part_ReasoningTrace has nil ReasoningTrace")
		}

		var reasoningMeta map[string]any
		if data.ReasoningTrace.Metadata != nil {
			reasoningMeta = data.ReasoningTrace.Metadata.AsMap()
		}

		part.ReasoningTrace = &llm.ReasoningTrace{
			ID:       data.ReasoningTrace.Id,
			Text:     data.ReasoningTrace.Text,
			Metadata: reasoningMeta,
		}

	case nil:
		return nil, errors.New("part has no data set")

	default:
		return nil, fmt.Errorf("unknown Part data type: %T", data)
	}

	// Convert part metadata
	if pb.Metadata != nil {
		part.Metadata = pb.Metadata.AsMap()
	}

	return part, nil
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

// toProtoPartKind converts llm.PartKind to proto enum.
func toProtoPartKind(kind llm.PartKind) (llmpb.PartKind, error) {
	switch kind {
	case llm.PartText:
		return llmpb.PartKind_PART_KIND_TEXT, nil
	case llm.PartToolRequest:
		return llmpb.PartKind_PART_KIND_TOOL_REQUEST, nil
	case llm.PartToolResponse:
		return llmpb.PartKind_PART_KIND_TOOL_RESPONSE, nil
	case llm.PartReasoning:
		return llmpb.PartKind_PART_KIND_REASONING, nil
	default:
		return llmpb.PartKind_PART_KIND_UNSPECIFIED, fmt.Errorf("unknown PartKind: %v", kind)
	}
}

// fromProtoPartKind converts proto PartKind to llm.PartKind.
func fromProtoPartKind(pbKind llmpb.PartKind) (llm.PartKind, error) {
	switch pbKind {
	case llmpb.PartKind_PART_KIND_TEXT:
		return llm.PartText, nil
	case llmpb.PartKind_PART_KIND_TOOL_REQUEST:
		return llm.PartToolRequest, nil
	case llmpb.PartKind_PART_KIND_TOOL_RESPONSE:
		return llm.PartToolResponse, nil
	case llmpb.PartKind_PART_KIND_REASONING:
		return llm.PartReasoning, nil
	case llmpb.PartKind_PART_KIND_UNSPECIFIED:
		return 0, errors.New("unspecified PartKind")
	default:
		return 0, fmt.Errorf("unknown PartKind: %v", pbKind)
	}
}
