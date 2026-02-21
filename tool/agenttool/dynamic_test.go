package agenttool_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/tool"
	"github.com/redpanda-data/ai-sdk-go/tool/agenttool"
)

func TestNewDynamic(t *testing.T) {
	t.Parallel()

	models := []string{"gpt-4", "claude-sonnet"}

	t.Run("successful execution with fresh agent each time", func(t *testing.T) {
		t.Parallel()

		callCount := 0
		factory := func(_ context.Context, args json.RawMessage) (agent.Agent, json.RawMessage, error) {
			callCount++
			return &mockAgent{
				name:     "dynamic-agent",
				response: fmt.Sprintf("response-%d", callCount),
			}, args, nil
		}

		dt := agenttool.NewDynamic(models, factory)
		require.NotNil(t, dt)
		assert.Implements(t, (*tool.Tool)(nil), dt)

		// First invocation
		result1, err := dt.Execute(context.Background(), json.RawMessage(`{"system_prompt":"be helpful","model":"gpt-4","message":"first"}`))
		require.NoError(t, err)

		var out1 agenttool.Result
		require.NoError(t, json.Unmarshal(result1, &out1))
		assert.Equal(t, "response-1", out1.Result)

		// Second invocation - factory called again, fresh agent
		result2, err := dt.Execute(context.Background(), json.RawMessage(`{"system_prompt":"be helpful","model":"gpt-4","message":"second"}`))
		require.NoError(t, err)

		var out2 agenttool.Result
		require.NoError(t, json.Unmarshal(result2, &out2))
		assert.Equal(t, "response-2", out2.Result)

		assert.Equal(t, 2, callCount)
	})

	t.Run("factory error", func(t *testing.T) {
		t.Parallel()

		factory := func(_ context.Context, _ json.RawMessage) (agent.Agent, json.RawMessage, error) {
			return nil, nil, errors.New("cannot create agent")
		}

		dt := agenttool.NewDynamic(models, factory)
		_, err := dt.Execute(context.Background(), json.RawMessage(`{"system_prompt":"x","model":"gpt-4","message":"hi"}`))

		require.Error(t, err)
		assert.Contains(t, err.Error(), "agent factory failed")
		assert.Contains(t, err.Error(), "cannot create agent")
	})

	t.Run("factory receives args and extracts message", func(t *testing.T) {
		t.Parallel()

		factory := func(_ context.Context, args json.RawMessage) (agent.Agent, json.RawMessage, error) {
			var params struct {
				SystemPrompt string `json:"system_prompt"`
				Model        string `json:"model"`
				Message      string `json:"message"`
			}
			require.NoError(t, json.Unmarshal(args, &params))
			assert.Equal(t, "gpt-4", params.Model)
			assert.Equal(t, "be concise", params.SystemPrompt)

			userMsg, err := json.Marshal(map[string]string{"message": params.Message})
			require.NoError(t, err)

			return &mockAgent{
				name:     "dynamic-agent",
				response: fmt.Sprintf("processed: %s", params.Message),
			}, userMsg, nil
		}

		dt := agenttool.NewDynamic(models, factory)
		result, err := dt.Execute(context.Background(), json.RawMessage(`{"system_prompt":"be concise","model":"gpt-4","message":"hello"}`))
		require.NoError(t, err)

		var out agenttool.Result
		require.NoError(t, json.Unmarshal(result, &out))
		assert.Equal(t, "processed: hello", out.Result)
	})

	t.Run("definition has fixed name and schema with model enum", func(t *testing.T) {
		t.Parallel()

		factory := func(_ context.Context, _ json.RawMessage) (agent.Agent, json.RawMessage, error) {
			t.Fatal("factory should not be called for Definition()")
			return nil, nil, nil
		}

		dt := agenttool.NewDynamic(models, factory)
		def := dt.Definition()

		assert.Equal(t, "dynamic_subagent", def.Name)
		assert.NotEmpty(t, def.Description)

		var schema map[string]any
		require.NoError(t, json.Unmarshal(def.Parameters, &schema))
		assert.Equal(t, "object", schema["type"])

		props, ok := schema["properties"].(map[string]any)
		require.True(t, ok)
		assert.Contains(t, props, "system_prompt")
		assert.Contains(t, props, "model")
		assert.Contains(t, props, "message")

		// Verify model enum
		modelProp := props["model"].(map[string]any)
		modelEnumRaw := modelProp["enum"].([]any)
		assert.ElementsMatch(t, []any{"gpt-4", "claude-sonnet"}, modelEnumRaw)
	})
}
