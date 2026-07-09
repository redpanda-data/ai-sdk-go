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

// ---------- InferenceProfileRegion ----------

func TestInferenceProfileRegion(t *testing.T) {
	t.Parallel()

	tests := []struct {
		region string
		want   string
	}{
		{"us-east-1", "us"},
		{"us-west-2", "us"},
		{"eu-west-1", "eu"},
		{"eu-central-1", "eu"},
		{"ap-southeast-1", "apac"},
		{"ap-northeast-1", "apac"},
		{"ap-south-1", "apac"},
		{"", "us"},
		{"unknown", "us"},
	}

	for _, tt := range tests {
		t.Run(tt.region, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, InferenceProfileRegion(tt.region))
		})
	}
}

// ---------- hasRegionPrefix ----------

func TestHasRegionPrefix(t *testing.T) {
	t.Parallel()

	tests := []struct {
		modelID string
		want    bool
	}{
		{"anthropic.claude-sonnet-4-6", false},
		{"us.anthropic.claude-sonnet-4-6", true},
		{"eu.anthropic.claude-sonnet-4-6", true},
		{"apac.anthropic.claude-sonnet-4-6", true},
		{"global.anthropic.claude-sonnet-4-6", true},
		{"no-dots-here", false},
		// Vendor-namespaced IDs whose version carries its own dot must not be
		// mistaken for region-prefixed (the dot-counting bug). "openai" is not
		// a geo prefix, so these are bare.
		{"openai.gpt-5.5", false},
		{"openai.gpt-5.4", false},
		{"openai.gpt-oss-120b-1:0", false},
		// A genuine geo prefix in front of such an ID is still detected.
		{"us.openai.gpt-5.5", true},
		{"eu.openai.gpt-5.5", true},
	}

	for _, tt := range tests {
		t.Run(tt.modelID, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, hasRegionPrefix(tt.modelID))
		})
	}
}

// ---------- lookupModel ----------

func TestLookupModel(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		input   string
		wantOK  bool
		wantDef string // expected ModelDefinition.Name if found
	}{
		{
			// Every Anthropic 4.5+ model on Bedrock returns
			// "ValidationException: on-demand throughput isn't supported"
			// for the bare ID. Verified empirically against bedrock-runtime
			// in us-east-2; we do not register bare entries for any of
			// them. See note at top of models.go.
			name:   "bare ID is not in the catalog for inference-profile-only models",
			input:  ModelClaudeSonnet45,
			wantOK: false,
		},
		{
			name:   "bare ID for 4.6 is also not in the catalog",
			input:  ModelClaudeSonnet46,
			wantOK: false,
		},
		{
			name:   "Sonnet 5 bare ID is not in the catalog",
			input:  ModelClaudeSonnet5,
			wantOK: false,
		},
		{
			name:    "Sonnet 5 US profile is its own entry",
			input:   ModelClaudeSonnet5US,
			wantOK:  true,
			wantDef: ModelClaudeSonnet5US,
		},
		{
			name:    "Sonnet 5 global profile is its own entry",
			input:   ModelClaudeSonnet5Global,
			wantOK:  true,
			wantDef: ModelClaudeSonnet5Global,
		},
		{
			name:   "Sonnet 5 eu profile is not published yet",
			input:  "eu." + ModelClaudeSonnet5,
			wantOK: false,
		},
		{
			// The Bedrock catalog registers inference-profile variants so
			// pricing and routing metadata stay explicit per profile.
			name:   "Fable 5 bare ID is not in the catalog",
			input:  ModelClaudeFable5,
			wantOK: false,
		},
		{
			name:    "geo profile is its own entry",
			input:   ModelClaudeSonnet46EU,
			wantOK:  true,
			wantDef: ModelClaudeSonnet46EU,
		},
		{
			name:    "Fable 5 US profile is its own entry",
			input:   ModelClaudeFable5US,
			wantOK:  true,
			wantDef: ModelClaudeFable5US,
		},
		{
			name:    "Fable 5 EU profile is its own entry",
			input:   ModelClaudeFable5EU,
			wantOK:  true,
			wantDef: ModelClaudeFable5EU,
		},
		{
			name:    "Fable 5 global profile is its own entry",
			input:   ModelClaudeFable5Global,
			wantOK:  true,
			wantDef: ModelClaudeFable5Global,
		},
		{
			name:    "global profile is its own entry",
			input:   ModelClaudeOpus47Global,
			wantOK:  true,
			wantDef: ModelClaudeOpus47Global,
		},
		{
			name:    "versioned model with region",
			input:   ModelClaudeHaiku45US,
			wantOK:  true,
			wantDef: ModelClaudeHaiku45US,
		},
		{
			name:   "unknown model",
			input:  "llama-3.2-90b",
			wantOK: false,
		},
		{
			name:   "unknown with region prefix",
			input:  "us.meta.llama-3.2-90b",
			wantOK: false,
		},
		{
			name:   "geo profile not published for this model",
			input:  "au." + ModelClaudeOpus45,
			wantOK: false,
		},

		// Amazon Nova 2 Lite — inference-profile-only, like the Claude
		// entries. global/us/eu/jp are registered; the bare ID and the
		// unpublished au. profile are not.
		{
			name:    "Nova 2 Lite global profile is its own entry",
			input:   ModelNova2LiteGlobal,
			wantOK:  true,
			wantDef: ModelNova2LiteGlobal,
		},
		{
			name:    "Nova 2 Lite us profile is its own entry",
			input:   ModelNova2LiteUS,
			wantOK:  true,
			wantDef: ModelNova2LiteUS,
		},
		{
			name:    "Nova 2 Lite eu profile is its own entry",
			input:   ModelNova2LiteEU,
			wantOK:  true,
			wantDef: ModelNova2LiteEU,
		},
		{
			name:    "Nova 2 Lite jp profile is its own entry",
			input:   ModelNova2LiteJP,
			wantOK:  true,
			wantDef: ModelNova2LiteJP,
		},
		{
			name:   "Nova 2 Lite bare ID is not in the catalog",
			input:  ModelNova2Lite,
			wantOK: false,
		},
		{
			name:   "Nova 2 Lite au profile is not published",
			input:  "au." + ModelNova2Lite,
			wantOK: false,
		},

		// Mistral Large 3 — us. profile only for now. The bare ID and the
		// not-yet-registered eu. profile are absent.
		{
			name:    "Mistral Large 3 us profile is its own entry",
			input:   ModelMistralLarge3US,
			wantOK:  true,
			wantDef: ModelMistralLarge3US,
		},
		{
			name:   "Mistral Large 3 bare ID is not in the catalog",
			input:  ModelMistralLarge3,
			wantOK: false,
		},
		{
			name:   "Mistral Large 3 eu profile is not registered (US-only)",
			input:  "eu." + ModelMistralLarge3,
			wantOK: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			def, ok := lookupModel(tt.input)
			assert.Equal(t, tt.wantOK, ok)

			if tt.wantOK {
				assert.Equal(t, tt.wantDef, def.Name)
			}
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
		{"model constant", ModelClaudeSonnet46},
		{"with region prefix", "eu." + ModelClaudeSonnet46},
		{"opus with region", "global." + ModelClaudeOpus46},
		{"haiku with region", "eu." + ModelClaudeHaiku45},
		{"Fable 5 with region prefix", ModelClaudeFable5EU},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			model, err := p.NewModel(tt.modelName)
			require.NoError(t, err)
			require.NotNil(t, model)
			assert.Equal(t, tt.modelName, model.Name())
			assert.Equal(t, "aws.bedrock", model.Provider())
		})
	}
}

func TestNewModel_APACRegionPrefixHasNoMatchingModel(t *testing.T) {
	t.Parallel()

	// AWS does not publish an "apac." inference profile for any current
	// Anthropic Claude model — Sonnet 4.5 has "jp." for Japan, and other
	// Asia-Pacific regions have to use "global." instead. A provider in
	// ap-southeast-1 calling NewModel with a bare Claude ID therefore fails
	// at lookup; that's the correct behavior, exposed here as a regression
	// guard so anyone re-introducing apac. needs to confirm AWS publishes it.
	p := &Provider{client: nil, region: "ap-southeast-1"}

	_, err := p.NewModel(ModelClaudeHaiku45)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported Bedrock model")
}

func TestNewModel_USRegionPrefix(t *testing.T) {
	t.Parallel()

	p := &Provider{client: nil, region: "us-east-1"}

	model, err := p.NewModel(ModelClaudeSonnet46)
	require.NoError(t, err)

	m, ok := model.(*Model)
	require.True(t, ok)
	assert.Equal(t, "us."+ModelClaudeSonnet46, m.config.APIModelID)
}

func TestNewModel_EURegionPrefix(t *testing.T) {
	t.Parallel()

	p := &Provider{client: nil, region: "eu-west-1"}

	model, err := p.NewModel(ModelClaudeSonnet46)
	require.NoError(t, err)

	m, ok := model.(*Model)
	require.True(t, ok)
	assert.Equal(t, "eu."+ModelClaudeSonnet46, m.config.APIModelID)
}

func TestNewModel_Fable5RegionPrefix(t *testing.T) {
	t.Parallel()

	p := &Provider{client: nil, region: "us-east-1"}

	model, err := p.NewModel(ModelClaudeFable5)
	require.NoError(t, err)

	m, ok := model.(*Model)
	require.True(t, ok)
	assert.Equal(t, ModelClaudeFable5US, m.config.APIModelID)
}

func TestNewModel_Fable5SamplingParametersRejected(t *testing.T) {
	t.Parallel()

	p := &Provider{client: nil, region: "us-east-1"}

	tests := []struct {
		name string
		opt  Option
		want string
	}{
		{name: "temperature", opt: WithTemperature(0.5), want: "temperature"},
		{name: "top_p", opt: WithTopP(0.9), want: "top_p"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			_, err := p.NewModel(ModelClaudeFable5, tt.opt)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.want)
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

	model, err := p.NewModel(ModelClaudeSonnet46,
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

	_, err := p.NewModel(ModelClaudeSonnet46, WithTemperature(2.0))
	require.Error(t, err)
}

func TestNewModel_MaxTokensExceedsLimit(t *testing.T) {
	t.Parallel()

	p := &Provider{client: nil}

	_, err := p.NewModel(ModelClaudeSonnet46, WithMaxTokens(999999))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "exceeds limit")
}

func TestNewModel_Capabilities(t *testing.T) {
	t.Parallel()

	p := &Provider{client: nil}

	model, err := p.NewModel(ModelClaudeOpus46)
	require.NoError(t, err)

	caps := model.Capabilities()
	assert.True(t, caps.Streaming)
	assert.True(t, caps.Tools)
	assert.False(t, caps.Vision)
	assert.True(t, caps.MultiTurn)
	assert.True(t, caps.SystemPrompts)
	assert.True(t, caps.Reasoning)
}

// ---------- Request mapper ----------

func TestRequestMapper_BasicRequest(t *testing.T) {
	t.Parallel()

	cfg := &Config{
		ModelName:  "eu.anthropic.claude-sonnet-4-6",
		APIModelID: "eu.anthropic.claude-sonnet-4-6",
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
		ModelName:   ModelClaudeSonnet46,
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
		ModelName:  ModelClaudeSonnet46,
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
		ModelName:  ModelClaudeSonnet46,
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
		ModelName:  ModelClaudeSonnet46,
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
		ModelName:  ModelClaudeSonnet46,
		setOptions: make(map[string]bool),
	}

	mapper := NewRequestMapper(cfg)

	req := &llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser,
				&llm.ToolResponsePart{
					ID:     "toolu_123",
					Name:   "search",
					Result: json.RawMessage(`{"results": ["cat1", "cat2"]}`),
				},
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
		ModelName:  ModelClaudeSonnet46,
		setOptions: make(map[string]bool),
	}

	mapper := NewRequestMapper(cfg)

	req := &llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser,
				&llm.ToolResponsePart{
					ID:      "toolu_123",
					Name:    "search",
					IsError: true, Result: json.RawMessage(`{"error":"API rate limited"}`),
				},
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
		ModelName:  ModelClaudeSonnet46,
		setOptions: make(map[string]bool),
	}

	mapper := NewRequestMapper(cfg)

	req := &llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleAssistant,
				llm.NewTextPart("Let me search for that."),
				&llm.ToolRequestPart{
					ID:        "toolu_456",
					Name:      "search",
					Arguments: json.RawMessage(`{"query":"cats"}`),
				},
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
		ModelName:     ModelClaudeSonnet46,
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
		ModelName:     ModelClaudeSonnet46,
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
		ModelName:   ModelClaudeSonnet46,
		APIModelID:  "anthropic.claude-sonnet-4-6",
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

	assert.Equal(t, "anthropic.claude-sonnet-4-6", *input.ModelId)
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

	resp, err := mapper.FromConverseOutput(
		types.StopReasonEndTurn,
		output,
		&types.TokenUsage{
			InputTokens:  aws.Int32(10),
			OutputTokens: aws.Int32(8),
			TotalTokens:  aws.Int32(18),
		},
		&types.PerformanceConfiguration{
			Latency: types.PerformanceConfigLatencyOptimized,
		},
		&types.ServiceTier{
			Type: types.ServiceTierTypeReserved,
		},
		&types.ConverseTrace{
			PromptRouter: &types.PromptRouterTrace{
				InvokedModelId: aws.String("us.anthropic.claude-sonnet-4-6"),
			},
		},
	)
	require.NoError(t, err)

	assert.Equal(t, llm.RoleAssistant, resp.Message.Role)
	assert.Equal(t, "Hello! How can I help?", resp.TextContent())
	assert.Equal(t, llm.FinishReasonStop, resp.FinishReason)
	assert.Equal(t, llm.ServiceTierReserved, resp.ServiceTier)
	// Bedrock reports "optimized"; NormalizeSpeed collapses it to the
	// cross-provider SpeedFast concept.
	assert.Equal(t, llm.SpeedFast, resp.Speed)
	// InvokedModelID reflects the actual inference profile AWS routed to,
	// not just the logical model name — geo and global profiles bill at
	// different rates so the routing identity matters downstream.
	assert.Equal(t, ModelClaudeSonnet46US, resp.InvokedModelID)
	require.NotNil(t, resp.Usage)
	assert.Equal(t, 10, resp.Usage.InputTokens)
	assert.Equal(t, 8, resp.Usage.OutputTokens)
	assert.Equal(t, 18, resp.Usage.TotalBilledTokens())
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
	}, nil, nil, nil)
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

	_, err := mapper.FromConverseOutput(types.StopReasonEndTurn, nil, nil, nil, nil, nil)
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
		InputTokens:          aws.Int32(100),
		OutputTokens:         aws.Int32(10),
		TotalTokens:          aws.Int32(110),
		CacheReadInputTokens: aws.Int32(80),
	}, nil, nil, nil)
	require.NoError(t, err)
	require.NotNil(t, resp.Usage)
	assert.Equal(t, 80, resp.Usage.CachedInputTokens)
}

// ---------- Models discovery ----------

func TestModelsDiscovery(t *testing.T) {
	t.Parallel()

	p := &Provider{}

	models := p.Models()
	assert.Len(t, models, len(supportedModels))

	for _, m := range models {
		assert.Equal(t, "aws.bedrock", m.Provider)
		assert.NotEmpty(t, m.Name)
		assert.NotEmpty(t, m.Label)
		assert.Positive(t, m.Constraints.MaxInputTokens,
			"model %s missing MaxInputTokens — set Constraints in its ModelDefinition", m.Name)
		assert.Positive(t, m.Constraints.MaxOutputTokens,
			"model %s missing MaxOutputTokens — set Constraints in its ModelDefinition", m.Name)
	}

	// Verify sorted by name
	for i := 1; i < len(models); i++ {
		assert.Less(t, models[i-1].Name, models[i].Name,
			"Models() should be sorted by Name: %s should come before %s", models[i-1].Name, models[i].Name)
	}
}

func TestModelsDiscovery_ProviderDataSharingMetadata(t *testing.T) {
	t.Parallel()

	p := &Provider{}

	models := p.Models()

	metadataByName := make(map[string]map[string]string, len(models))
	for _, m := range models {
		metadataByName[m.Name] = m.Metadata
	}

	for _, name := range []string{ModelClaudeFable5Global, ModelClaudeFable5US, ModelClaudeFable5EU} {
		assert.Equal(t, "true", metadataByName[name][ModelMetadataRequiresProviderDataSharing])
	}

	assert.Empty(t, metadataByName[ModelClaudeSonnet46US])
}

// ---------- Options validation ----------

func TestWithStop_TooMany(t *testing.T) {
	t.Parallel()

	p := &Provider{client: nil}

	_, err := p.NewModel(ModelClaudeSonnet46, WithStop("a", "b", "c", "d", "e"))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "maximum 4 stop sequences")
}

func TestWithStop_Empty(t *testing.T) {
	t.Parallel()

	p := &Provider{client: nil}

	_, err := p.NewModel(ModelClaudeSonnet46, WithStop("a", ""))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cannot be empty")
}

func TestWithTopP_OutOfRange(t *testing.T) {
	t.Parallel()

	p := &Provider{client: nil}

	_, err := p.NewModel(ModelClaudeSonnet46, WithTopP(1.5))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "top_p must be 0.0-1.0")
}

func TestWithMaxTokens_Negative(t *testing.T) {
	t.Parallel()

	p := &Provider{client: nil}

	_, err := p.NewModel(ModelClaudeSonnet46, WithMaxTokens(-1))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "must be positive")
}
