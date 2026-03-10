package bedrock

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// ---------- resolveModelFamily ----------

func TestResolveModelFamily(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    string
		expected string
	}{
		// Direct family names
		{
			name:     "direct family: claude-sonnet-4-6",
			input:    "claude-sonnet-4-6",
			expected: "claude-sonnet-4-6",
		},
		{
			name:     "direct family: claude-opus-4-6",
			input:    "claude-opus-4-6",
			expected: "claude-opus-4-6",
		},
		{
			name:     "direct family: claude-haiku-4-5",
			input:    "claude-haiku-4-5",
			expected: "claude-haiku-4-5",
		},
		// Cross-region inference profiles
		{
			name:     "eu cross-region: eu.anthropic.claude-sonnet-4-6",
			input:    "eu.anthropic.claude-sonnet-4-6",
			expected: "claude-sonnet-4-6",
		},
		{
			name:     "global cross-region: global.anthropic.claude-opus-4-6-v1",
			input:    "global.anthropic.claude-opus-4-6-v1",
			expected: "claude-opus-4-6",
		},
		// Versioned inference profiles
		{
			name:     "versioned: eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
			input:    "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
			expected: "claude-sonnet-4-5",
		},
		{
			name:     "versioned: eu.anthropic.claude-haiku-4-5-20251001-v1:0",
			input:    "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
			expected: "claude-haiku-4-5",
		},
		{
			name:     "versioned: global.anthropic.claude-opus-4-5-20251101-v1:0",
			input:    "global.anthropic.claude-opus-4-5-20251101-v1:0",
			expected: "claude-opus-4-5",
		},
		// Provider prefix without region
		{
			name:     "provider prefix: anthropic.claude-sonnet-4-6-v2:0",
			input:    "anthropic.claude-sonnet-4-6-v2:0",
			expected: "claude-sonnet-4-6",
		},
		// Unknown model — returned as-is
		{
			name:     "unknown model returned as-is",
			input:    "eu.anthropic.claude-3-5-sonnet-20240620-v1:0",
			expected: "eu.anthropic.claude-3-5-sonnet-20240620-v1:0",
		},
		{
			name:     "completely unknown model",
			input:    "llama-3.2-90b",
			expected: "llama-3.2-90b",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.expected, resolveModelFamily(tt.input))
		})
	}
}

// ---------- classifyError ----------

func TestClassifyError_Nil(t *testing.T) {
	t.Parallel()
	assert.NoError(t, classifyError(nil))
}

func TestClassifyError_UnknownError(t *testing.T) {
	t.Parallel()

	err := errors.New("something unexpected")
	result := classifyError(err)
	assert.Equal(t, err, result)
}

func TestClassifyError_AWSExceptions(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		err       error
		wantBase  error
		wantRetry bool
		wantCode  string
	}{
		{
			name:      "ThrottlingException",
			err:       &types.ThrottlingException{Message: aws.String("Rate exceeded")},
			wantBase:  llm.ErrRateLimitExceeded,
			wantRetry: true,
			wantCode:  "ThrottlingException",
		},
		{
			name:      "ServiceQuotaExceededException",
			err:       &types.ServiceQuotaExceededException{Message: aws.String("Quota exceeded")},
			wantBase:  llm.ErrRateLimitExceeded,
			wantRetry: true,
			wantCode:  "ServiceQuotaExceededException",
		},
		{
			name:      "ValidationException",
			err:       &types.ValidationException{Message: aws.String("Invalid input")},
			wantBase:  llm.ErrInvalidInput,
			wantRetry: false,
			wantCode:  "ValidationException",
		},
		{
			name:      "AccessDeniedException",
			err:       &types.AccessDeniedException{Message: aws.String("Access denied")},
			wantBase:  llm.ErrAPICall,
			wantRetry: false,
			wantCode:  "AccessDeniedException",
		},
		{
			name:      "ResourceNotFoundException",
			err:       &types.ResourceNotFoundException{Message: aws.String("Model not found")},
			wantBase:  llm.ErrAPICall,
			wantRetry: false,
			wantCode:  "ResourceNotFoundException",
		},
		{
			name:      "ModelTimeoutException",
			err:       &types.ModelTimeoutException{Message: aws.String("Timeout")},
			wantBase:  llm.ErrServerError,
			wantRetry: true,
			wantCode:  "ModelTimeoutException",
		},
		{
			name:      "InternalServerException",
			err:       &types.InternalServerException{Message: aws.String("Internal error")},
			wantBase:  llm.ErrServerError,
			wantRetry: true,
			wantCode:  "InternalServerException",
		},
		{
			name:      "ServiceUnavailableException",
			err:       &types.ServiceUnavailableException{Message: aws.String("Unavailable")},
			wantBase:  llm.ErrServerError,
			wantRetry: true,
			wantCode:  "ServiceUnavailableException",
		},
		{
			name:      "ModelErrorException",
			err:       &types.ModelErrorException{Message: aws.String("Model error")},
			wantBase:  llm.ErrServerError,
			wantRetry: true,
			wantCode:  "ModelErrorException",
		},
		{
			name:      "ModelStreamErrorException",
			err:       &types.ModelStreamErrorException{Message: aws.String("Stream error")},
			wantBase:  llm.ErrServerError,
			wantRetry: true,
			wantCode:  "ModelStreamErrorException",
		},
		{
			name:      "ModelNotReadyException",
			err:       &types.ModelNotReadyException{Message: aws.String("Not ready")},
			wantBase:  llm.ErrServerError,
			wantRetry: true,
			wantCode:  "ModelNotReadyException",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result := classifyError(tt.err)
			require.Error(t, result)

			var pe *llm.ProviderError
			require.ErrorAs(t, result, &pe)
			require.ErrorIs(t, pe, tt.wantBase)
			assert.Equal(t, tt.wantRetry, pe.Retryable)
			assert.Equal(t, tt.wantCode, pe.Code)
		})
	}
}

// ---------- NewModel validation ----------

func TestNewModel_SupportedModels(t *testing.T) {
	t.Parallel()

	// Create a provider with a pre-loaded AWS config to avoid AWS credential lookup
	p := &Provider{client: nil}

	tests := []struct {
		name      string
		modelName string
	}{
		{"direct family", "claude-sonnet-4-6"},
		{"eu cross-region", "eu.anthropic.claude-sonnet-4-6"},
		{"global versioned", "global.anthropic.claude-opus-4-6-v1"},
		{"versioned with suffix", "eu.anthropic.claude-haiku-4-5-20251001-v1:0"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			model, err := p.NewModel(tt.modelName)
			require.NoError(t, err)
			require.NotNil(t, model)
			assert.Equal(t, tt.modelName, model.Name())
			assert.Equal(t, "bedrock", model.Provider())
		})
	}
}

func TestNewModel_UnsupportedModel(t *testing.T) {
	t.Parallel()

	p := &Provider{client: nil}

	_, err := p.NewModel("llama-3.2-90b")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported Bedrock model")
}

func TestNewModel_WithOptions(t *testing.T) {
	t.Parallel()

	p := &Provider{client: nil}

	model, err := p.NewModel("claude-sonnet-4-6",
		WithTemperature(0.7),
		WithTopP(0.9),
		WithMaxTokens(1000),
		WithStop("END", "STOP"),
	)
	require.NoError(t, err)

	m, ok := model.(*Model)
	require.True(t, ok)
	require.NotNil(t, m.config.Temperature)
	assert.InDelta(t, 0.7, *m.config.Temperature, 0.001)
	require.NotNil(t, m.config.TopP)
	assert.InDelta(t, 0.9, *m.config.TopP, 0.001)
	require.NotNil(t, m.config.MaxTokens)
	assert.Equal(t, int32(1000), *m.config.MaxTokens)
	assert.Equal(t, []string{"END", "STOP"}, m.config.Stop)
}

func TestNewModel_InvalidTemperature(t *testing.T) {
	t.Parallel()

	p := &Provider{client: nil}

	_, err := p.NewModel("claude-sonnet-4-6", WithTemperature(2.0))
	require.Error(t, err)
}

func TestNewModel_MaxTokensExceedsLimit(t *testing.T) {
	t.Parallel()

	p := &Provider{client: nil}

	_, err := p.NewModel("claude-sonnet-4-6", WithMaxTokens(999999))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "exceeds limit")
}

func TestNewModel_Capabilities(t *testing.T) {
	t.Parallel()

	p := &Provider{client: nil}

	model, err := p.NewModel("claude-opus-4-6")
	require.NoError(t, err)

	caps := model.Capabilities()
	assert.True(t, caps.Streaming)
	assert.True(t, caps.Tools)
	assert.True(t, caps.Vision)
	assert.True(t, caps.MultiTurn)
	assert.True(t, caps.SystemPrompts)
	assert.True(t, caps.Reasoning)
}

// ---------- Request mapper ----------

func TestRequestMapper_BasicRequest(t *testing.T) {
	t.Parallel()

	cfg := &Config{
		ModelName:  "eu.anthropic.claude-sonnet-4-6",
		setOptions: make(map[string]bool),
	}

	mapper := NewRequestMapper(cfg)

	req := &llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleSystem, llm.NewTextPart("You are helpful.")),
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hello!")),
		},
	}

	input, err := mapper.ToConverseInput(req)
	require.NoError(t, err)

	assert.Equal(t, "eu.anthropic.claude-sonnet-4-6", *input.ModelId)
	assert.Len(t, input.System, 1)
	assert.Len(t, input.Messages, 1)

	// Check system content
	sysBlock, ok := input.System[0].(*types.SystemContentBlockMemberText)
	require.True(t, ok)
	assert.Equal(t, "You are helpful.", sysBlock.Value)

	// Check user message
	assert.Equal(t, types.ConversationRoleUser, input.Messages[0].Role)
	textBlock, ok := input.Messages[0].Content[0].(*types.ContentBlockMemberText)
	require.True(t, ok)
	assert.Equal(t, "Hello!", textBlock.Value)
}

func TestRequestMapper_InferenceConfig(t *testing.T) {
	t.Parallel()

	temp := 0.5
	topP := 0.8
	maxTokens := int32(2048)

	cfg := &Config{
		ModelName:  "claude-sonnet-4-6",
		Temperature: &temp,
		TopP:        &topP,
		MaxTokens:   &maxTokens,
		Stop:        []string{"END"},
		setOptions:  make(map[string]bool),
	}

	mapper := NewRequestMapper(cfg)

	input, err := mapper.ToConverseInput(&llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hi")),
		},
	})
	require.NoError(t, err)

	require.NotNil(t, input.InferenceConfig)
	require.NotNil(t, input.InferenceConfig.Temperature)
	assert.InDelta(t, 0.5, *input.InferenceConfig.Temperature, 0.001)
	require.NotNil(t, input.InferenceConfig.TopP)
	assert.InDelta(t, 0.8, *input.InferenceConfig.TopP, 0.001)
	require.NotNil(t, input.InferenceConfig.MaxTokens)
	assert.Equal(t, int32(2048), *input.InferenceConfig.MaxTokens)
	assert.Equal(t, []string{"END"}, input.InferenceConfig.StopSequences)
}

func TestRequestMapper_NoInferenceConfig(t *testing.T) {
	t.Parallel()

	cfg := &Config{
		ModelName:  "claude-sonnet-4-6",
		setOptions: make(map[string]bool),
	}

	mapper := NewRequestMapper(cfg)

	input, err := mapper.ToConverseInput(&llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hi")),
		},
	})
	require.NoError(t, err)
	assert.Nil(t, input.InferenceConfig)
}

func TestRequestMapper_ToolDefinitions(t *testing.T) {
	t.Parallel()

	cfg := &Config{
		ModelName:  "claude-sonnet-4-6",
		setOptions: make(map[string]bool),
	}

	mapper := NewRequestMapper(cfg)

	schema := json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}`)

	req := &llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("Search for cats")),
		},
		Tools: []llm.ToolDefinition{
			{
				Name:        "search",
				Description: "Search the web",
				Parameters:  schema,
			},
		},
		ToolChoice: &llm.ToolChoice{Type: llm.ToolChoiceAuto},
	}

	input, err := mapper.ToConverseInput(req)
	require.NoError(t, err)

	require.NotNil(t, input.ToolConfig)
	assert.Len(t, input.ToolConfig.Tools, 1)

	toolSpec, ok := input.ToolConfig.Tools[0].(*types.ToolMemberToolSpec)
	require.True(t, ok)
	assert.Equal(t, "search", *toolSpec.Value.Name)
	assert.Equal(t, "Search the web", *toolSpec.Value.Description)

	// Check tool choice
	_, ok = input.ToolConfig.ToolChoice.(*types.ToolChoiceMemberAuto)
	assert.True(t, ok)
}

func TestRequestMapper_ToolChoiceSpecific(t *testing.T) {
	t.Parallel()

	cfg := &Config{
		ModelName:  "claude-sonnet-4-6",
		setOptions: make(map[string]bool),
	}

	mapper := NewRequestMapper(cfg)
	toolName := "search"

	req := &llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("Search")),
		},
		Tools: []llm.ToolDefinition{
			{
				Name:        "search",
				Description: "Search",
				Parameters:  json.RawMessage(`{"type":"object"}`),
			},
		},
		ToolChoice: &llm.ToolChoice{Type: llm.ToolChoiceSpecific, Name: &toolName},
	}

	input, err := mapper.ToConverseInput(req)
	require.NoError(t, err)

	tc, ok := input.ToolConfig.ToolChoice.(*types.ToolChoiceMemberTool)
	require.True(t, ok)
	assert.Equal(t, "search", *tc.Value.Name)
}

func TestRequestMapper_ToolResponse(t *testing.T) {
	t.Parallel()

	cfg := &Config{
		ModelName:  "claude-sonnet-4-6",
		setOptions: make(map[string]bool),
	}

	mapper := NewRequestMapper(cfg)

	req := &llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser,
				llm.NewToolResponsePart(&llm.ToolResponse{
					ID:     "toolu_123",
					Name:   "search",
					Result: json.RawMessage(`{"results": ["cat1", "cat2"]}`),
				}),
			),
		},
	}

	input, err := mapper.ToConverseInput(req)
	require.NoError(t, err)

	require.Len(t, input.Messages, 1)
	require.Len(t, input.Messages[0].Content, 1)

	toolResult, ok := input.Messages[0].Content[0].(*types.ContentBlockMemberToolResult)
	require.True(t, ok)
	assert.Equal(t, "toolu_123", *toolResult.Value.ToolUseId)
	assert.Equal(t, types.ToolResultStatusSuccess, toolResult.Value.Status)
}

func TestRequestMapper_ToolResponseError(t *testing.T) {
	t.Parallel()

	cfg := &Config{
		ModelName:  "claude-sonnet-4-6",
		setOptions: make(map[string]bool),
	}

	mapper := NewRequestMapper(cfg)

	req := &llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser,
				llm.NewToolResponsePart(&llm.ToolResponse{
					ID:    "toolu_123",
					Name:  "search",
					Error: "API rate limited",
				}),
			),
		},
	}

	input, err := mapper.ToConverseInput(req)
	require.NoError(t, err)

	toolResult, ok := input.Messages[0].Content[0].(*types.ContentBlockMemberToolResult)
	require.True(t, ok)
	assert.Equal(t, types.ToolResultStatusError, toolResult.Value.Status)
}

func TestRequestMapper_AssistantWithToolUse(t *testing.T) {
	t.Parallel()

	cfg := &Config{
		ModelName:  "claude-sonnet-4-6",
		setOptions: make(map[string]bool),
	}

	mapper := NewRequestMapper(cfg)

	req := &llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleAssistant,
				llm.NewTextPart("Let me search for that."),
				llm.NewToolRequestPart(&llm.ToolRequest{
					ID:        "toolu_456",
					Name:      "search",
					Arguments: json.RawMessage(`{"query":"cats"}`),
				}),
			),
		},
	}

	input, err := mapper.ToConverseInput(req)
	require.NoError(t, err)

	require.Len(t, input.Messages, 1)
	assert.Equal(t, types.ConversationRoleAssistant, input.Messages[0].Role)
	require.Len(t, input.Messages[0].Content, 2)

	// Text block
	textBlock, ok := input.Messages[0].Content[0].(*types.ContentBlockMemberText)
	require.True(t, ok)
	assert.Equal(t, "Let me search for that.", textBlock.Value)

	// Tool use block
	toolBlock, ok := input.Messages[0].Content[1].(*types.ContentBlockMemberToolUse)
	require.True(t, ok)
	assert.Equal(t, "toolu_456", *toolBlock.Value.ToolUseId)
	assert.Equal(t, "search", *toolBlock.Value.Name)
}

func TestRequestMapper_CachingEnabled(t *testing.T) {
	t.Parallel()

	cfg := &Config{
		ModelName:     "claude-sonnet-4-6",
		EnableCaching: true,
		setOptions:    make(map[string]bool),
	}

	mapper := NewRequestMapper(cfg)

	req := &llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleSystem, llm.NewTextPart("You are helpful.")),
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hello!")),
		},
	}

	input, err := mapper.ToConverseInput(req)
	require.NoError(t, err)

	// System should have the text block + a CachePoint
	require.Len(t, input.System, 2)
	_, ok := input.System[0].(*types.SystemContentBlockMemberText)
	require.True(t, ok)
	cacheBlock, ok := input.System[1].(*types.SystemContentBlockMemberCachePoint)
	require.True(t, ok)
	assert.Equal(t, types.CachePointTypeDefault, cacheBlock.Value.Type)

	// Last message should have text block + a CachePoint
	require.Len(t, input.Messages, 1)
	require.Len(t, input.Messages[0].Content, 2)
	_, ok = input.Messages[0].Content[0].(*types.ContentBlockMemberText)
	require.True(t, ok)
	msgCacheBlock, ok := input.Messages[0].Content[1].(*types.ContentBlockMemberCachePoint)
	require.True(t, ok)
	assert.Equal(t, types.CachePointTypeDefault, msgCacheBlock.Value.Type)
}

func TestRequestMapper_CachingDisabled(t *testing.T) {
	t.Parallel()

	cfg := &Config{
		ModelName:     "claude-sonnet-4-6",
		EnableCaching: false,
		setOptions:    make(map[string]bool),
	}

	mapper := NewRequestMapper(cfg)

	req := &llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleSystem, llm.NewTextPart("You are helpful.")),
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hello!")),
		},
	}

	input, err := mapper.ToConverseInput(req)
	require.NoError(t, err)

	// No cache points should be appended
	assert.Len(t, input.System, 1)
	require.Len(t, input.Messages, 1)
	assert.Len(t, input.Messages[0].Content, 1)
}

func TestRequestMapper_StreamInput(t *testing.T) {
	t.Parallel()

	temp := 0.5
	cfg := &Config{
		ModelName:   "claude-sonnet-4-6",
		Temperature: &temp,
		setOptions:  make(map[string]bool),
	}

	mapper := NewRequestMapper(cfg)

	req := &llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hi")),
		},
	}

	input, err := mapper.ToConverseStreamInput(req)
	require.NoError(t, err)

	assert.Equal(t, "claude-sonnet-4-6", *input.ModelId)
	require.NotNil(t, input.InferenceConfig)
	require.NotNil(t, input.InferenceConfig.Temperature)
	assert.InDelta(t, 0.5, *input.InferenceConfig.Temperature, 0.001)
}

// ---------- Response mapper ----------

func TestResponseMapper_TextResponse(t *testing.T) {
	t.Parallel()

	mapper := NewResponseMapper(supportedModels[ModelClaudeSonnet46])

	output := &types.ConverseOutputMemberMessage{
		Value: types.Message{
			Role: types.ConversationRoleAssistant,
			Content: []types.ContentBlock{
				&types.ContentBlockMemberText{Value: "Hello! How can I help?"},
			},
		},
	}

	resp, err := mapper.FromConverseOutput(types.StopReasonEndTurn, output, &types.TokenUsage{
		InputTokens:  aws.Int32(10),
		OutputTokens: aws.Int32(8),
		TotalTokens:  aws.Int32(18),
	})
	require.NoError(t, err)

	assert.Equal(t, llm.RoleAssistant, resp.Message.Role)
	assert.Equal(t, "Hello! How can I help?", resp.TextContent())
	assert.Equal(t, llm.FinishReasonStop, resp.FinishReason)
	require.NotNil(t, resp.Usage)
	assert.Equal(t, 10, resp.Usage.InputTokens)
	assert.Equal(t, 8, resp.Usage.OutputTokens)
	assert.Equal(t, 18, resp.Usage.TotalTokens)
}

func TestResponseMapper_ToolUseResponse(t *testing.T) {
	t.Parallel()

	mapper := NewResponseMapper(supportedModels[ModelClaudeSonnet46])

	output := &types.ConverseOutputMemberMessage{
		Value: types.Message{
			Role: types.ConversationRoleAssistant,
			Content: []types.ContentBlock{
				&types.ContentBlockMemberText{Value: "Let me search."},
				&types.ContentBlockMemberToolUse{
					Value: types.ToolUseBlock{
						ToolUseId: aws.String("toolu_123"),
						Name:      aws.String("search"),
						Input:     document.NewLazyDocument(map[string]any{"query": "cats"}),
					},
				},
			},
		},
	}

	resp, err := mapper.FromConverseOutput(types.StopReasonToolUse, output, &types.TokenUsage{
		InputTokens:  aws.Int32(20),
		OutputTokens: aws.Int32(15),
		TotalTokens:  aws.Int32(35),
	})
	require.NoError(t, err)

	assert.Equal(t, llm.FinishReasonToolCalls, resp.FinishReason)
	assert.True(t, resp.HasToolRequests())

	toolReqs := resp.ToolRequests()
	require.Len(t, toolReqs, 1)
	assert.Equal(t, "toolu_123", toolReqs[0].ID)
	assert.Equal(t, "search", toolReqs[0].Name)
}

func TestResponseMapper_StopReasons(t *testing.T) {
	t.Parallel()

	mapper := NewResponseMapper(supportedModels[ModelClaudeSonnet46])

	tests := []struct {
		name     string
		reason   types.StopReason
		expected llm.FinishReason
	}{
		{"end_turn", types.StopReasonEndTurn, llm.FinishReasonStop},
		{"stop_sequence", types.StopReasonStopSequence, llm.FinishReasonStop},
		{"max_tokens", types.StopReasonMaxTokens, llm.FinishReasonLength},
		{"content_filtered", types.StopReasonContentFiltered, llm.FinishReasonContentFilter},
		{"guardrail_intervened", types.StopReasonGuardrailIntervened, llm.FinishReasonContentFilter},
		{"tool_use", types.StopReasonToolUse, llm.FinishReasonToolCalls},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.expected, mapper.mapStopReason(tt.reason))
		})
	}
}

func TestResponseMapper_NilOutput(t *testing.T) {
	t.Parallel()

	mapper := NewResponseMapper(supportedModels[ModelClaudeSonnet46])

	_, err := mapper.FromConverseOutput(types.StopReasonEndTurn, nil, nil)
	require.Error(t, err)
	assert.ErrorIs(t, err, llm.ErrResponseMapping)
}

func TestResponseMapper_CachedTokens(t *testing.T) {
	t.Parallel()

	mapper := NewResponseMapper(supportedModels[ModelClaudeSonnet46])

	output := &types.ConverseOutputMemberMessage{
		Value: types.Message{
			Role: types.ConversationRoleAssistant,
			Content: []types.ContentBlock{
				&types.ContentBlockMemberText{Value: "Hi"},
			},
		},
	}

	resp, err := mapper.FromConverseOutput(types.StopReasonEndTurn, output, &types.TokenUsage{
		InputTokens:         aws.Int32(100),
		OutputTokens:        aws.Int32(10),
		TotalTokens:         aws.Int32(110),
		CacheReadInputTokens: aws.Int32(80),
	})
	require.NoError(t, err)
	require.NotNil(t, resp.Usage)
	assert.Equal(t, 80, resp.Usage.CachedTokens)
}

// ---------- Models discovery ----------

func TestModelsDiscovery(t *testing.T) {
	t.Parallel()

	p := &Provider{}

	models := p.Models()
	assert.Len(t, models, len(supportedModels))

	for _, m := range models {
		assert.Equal(t, "bedrock", m.Provider)
		assert.NotEmpty(t, m.Name)
		assert.NotEmpty(t, m.Label)
	}

	// Verify sorted by name
	for i := 1; i < len(models); i++ {
		assert.True(t, models[i-1].Name < models[i].Name,
			"Models() should be sorted by Name: %s should come before %s", models[i-1].Name, models[i].Name)
	}
}

// ---------- Options validation ----------

func TestWithStop_TooMany(t *testing.T) {
	t.Parallel()

	p := &Provider{client: nil}

	_, err := p.NewModel("claude-sonnet-4-6", WithStop("a", "b", "c", "d", "e"))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "maximum 4 stop sequences")
}

func TestWithStop_Empty(t *testing.T) {
	t.Parallel()

	p := &Provider{client: nil}

	_, err := p.NewModel("claude-sonnet-4-6", WithStop("a", ""))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cannot be empty")
}

func TestWithTopP_OutOfRange(t *testing.T) {
	t.Parallel()

	p := &Provider{client: nil}

	_, err := p.NewModel("claude-sonnet-4-6", WithTopP(1.5))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "top_p must be 0.0-1.0")
}

func TestWithMaxTokens_Negative(t *testing.T) {
	t.Parallel()

	p := &Provider{client: nil}

	_, err := p.NewModel("claude-sonnet-4-6", WithMaxTokens(-1))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "must be positive")
}
