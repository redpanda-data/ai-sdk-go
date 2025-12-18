package tool

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// simpleTool is a basic tool implementation for testing
type simpleTool struct {
	name        string
	description string
	result      string
}

func (t *simpleTool) Definition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Name:        t.name,
		Description: t.description,
		Parameters:  json.RawMessage(`{"type":"object","properties":{}}`),
	}
}

func (t *simpleTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	return json.Marshal(map[string]string{"result": t.result})
}

func TestCompositeRegistry_List(t *testing.T) {
	t.Parallel()

	t.Run("empty composite", func(t *testing.T) {
		t.Parallel()
		composite := NewCompositeRegistry()
		defs := composite.List()
		assert.Empty(t, defs)
	})

	t.Run("single child registry", func(t *testing.T) {
		t.Parallel()

		reg1 := NewRegistry(RegistryConfig{})
		err := reg1.Register(&simpleTool{name: "tool1", description: "Tool 1"})
		require.NoError(t, err)

		composite := NewCompositeRegistry(reg1)
		defs := composite.List()

		require.Len(t, defs, 1)
		assert.Equal(t, "tool1", defs[0].Name)
	})

	t.Run("multiple child registries", func(t *testing.T) {
		t.Parallel()

		reg1 := NewRegistry(RegistryConfig{})
		err := reg1.Register(&simpleTool{name: "tool1", description: "Tool 1"})
		require.NoError(t, err)

		reg2 := NewRegistry(RegistryConfig{})
		err = reg2.Register(&simpleTool{name: "tool2", description: "Tool 2"})
		require.NoError(t, err)

		composite := NewCompositeRegistry(reg1, reg2)
		defs := composite.List()

		require.Len(t, defs, 2)
		names := []string{defs[0].Name, defs[1].Name}
		assert.Contains(t, names, "tool1")
		assert.Contains(t, names, "tool2")
	})

	t.Run("duplicate tool names - first wins", func(t *testing.T) {
		t.Parallel()

		reg1 := NewRegistry(RegistryConfig{})
		err := reg1.Register(&simpleTool{name: "shared", description: "From reg1"})
		require.NoError(t, err)

		reg2 := NewRegistry(RegistryConfig{})
		err = reg2.Register(&simpleTool{name: "shared", description: "From reg2"})
		require.NoError(t, err)

		composite := NewCompositeRegistry(reg1, reg2)
		defs := composite.List()

		require.Len(t, defs, 1, "duplicate tool names should only appear once")
		assert.Equal(t, "shared", defs[0].Name)
		assert.Equal(t, "From reg1", defs[0].Description, "first registry wins for duplicates")
	})

	t.Run("mixed unique and duplicate tools", func(t *testing.T) {
		t.Parallel()

		reg1 := NewRegistry(RegistryConfig{})
		require.NoError(t, reg1.Register(&simpleTool{name: "tool1", description: "Tool 1"}))
		require.NoError(t, reg1.Register(&simpleTool{name: "shared", description: "From reg1"}))

		reg2 := NewRegistry(RegistryConfig{})
		require.NoError(t, reg2.Register(&simpleTool{name: "tool2", description: "Tool 2"}))
		require.NoError(t, reg2.Register(&simpleTool{name: "shared", description: "From reg2"}))

		composite := NewCompositeRegistry(reg1, reg2)
		defs := composite.List()

		require.Len(t, defs, 3, "should have 3 unique tool names")

		names := make(map[string]string)
		for _, def := range defs {
			names[def.Name] = def.Description
		}

		assert.Equal(t, "Tool 1", names["tool1"])
		assert.Equal(t, "Tool 2", names["tool2"])
		assert.Equal(t, "From reg1", names["shared"], "first registry wins for duplicates")
	})
}

func TestCompositeRegistry_Get(t *testing.T) {
	t.Parallel()

	t.Run("get from first registry", func(t *testing.T) {
		t.Parallel()

		reg1 := NewRegistry(RegistryConfig{})
		require.NoError(t, reg1.Register(&simpleTool{name: "tool1"}))

		reg2 := NewRegistry(RegistryConfig{})
		require.NoError(t, reg2.Register(&simpleTool{name: "tool2"}))

		composite := NewCompositeRegistry(reg1, reg2)

		tool, err := composite.Get("tool1")
		require.NoError(t, err)
		assert.Equal(t, "tool1", tool.Definition().Name)
	})

	t.Run("get from second registry", func(t *testing.T) {
		t.Parallel()

		reg1 := NewRegistry(RegistryConfig{})
		require.NoError(t, reg1.Register(&simpleTool{name: "tool1"}))

		reg2 := NewRegistry(RegistryConfig{})
		require.NoError(t, reg2.Register(&simpleTool{name: "tool2"}))

		composite := NewCompositeRegistry(reg1, reg2)

		tool, err := composite.Get("tool2")
		require.NoError(t, err)
		assert.Equal(t, "tool2", tool.Definition().Name)
	})

	t.Run("tool not found", func(t *testing.T) {
		t.Parallel()

		reg1 := NewRegistry(RegistryConfig{})
		require.NoError(t, reg1.Register(&simpleTool{name: "tool1"}))

		composite := NewCompositeRegistry(reg1)

		_, err := composite.Get("nonexistent")
		assert.ErrorIs(t, err, ErrToolNotFound)
	})

	t.Run("duplicate tool name - first registry wins", func(t *testing.T) {
		t.Parallel()

		reg1 := NewRegistry(RegistryConfig{})
		require.NoError(t, reg1.Register(&simpleTool{name: "shared", description: "From reg1"}))

		reg2 := NewRegistry(RegistryConfig{})
		require.NoError(t, reg2.Register(&simpleTool{name: "shared", description: "From reg2"}))

		composite := NewCompositeRegistry(reg1, reg2)

		tool, err := composite.Get("shared")
		require.NoError(t, err)
		assert.Equal(t, "From reg1", tool.Definition().Description)
	})
}

func TestCompositeRegistry_Execute(t *testing.T) {
	t.Parallel()

	t.Run("execute from first registry", func(t *testing.T) {
		t.Parallel()

		reg1 := NewRegistry(RegistryConfig{})
		require.NoError(t, reg1.Register(&simpleTool{name: "tool1", result: "result1"}))

		composite := NewCompositeRegistry(reg1)

		resp, err := composite.Execute(context.Background(), &llm.ToolRequest{
			ID:        "test-1",
			Name:      "tool1",
			Arguments: json.RawMessage(`{}`),
		})

		require.NoError(t, err)
		assert.Equal(t, "test-1", resp.ID)
		assert.Contains(t, string(resp.Result), "result1")
	})

	t.Run("execute from second registry", func(t *testing.T) {
		t.Parallel()

		reg1 := NewRegistry(RegistryConfig{})
		require.NoError(t, reg1.Register(&simpleTool{name: "tool1", result: "result1"}))

		reg2 := NewRegistry(RegistryConfig{})
		require.NoError(t, reg2.Register(&simpleTool{name: "tool2", result: "result2"}))

		composite := NewCompositeRegistry(reg1, reg2)

		resp, err := composite.Execute(context.Background(), &llm.ToolRequest{
			ID:        "test-2",
			Name:      "tool2",
			Arguments: json.RawMessage(`{}`),
		})

		require.NoError(t, err)
		assert.Equal(t, "test-2", resp.ID)
		assert.Contains(t, string(resp.Result), "result2")
	})

	t.Run("execute nonexistent tool", func(t *testing.T) {
		t.Parallel()

		reg1 := NewRegistry(RegistryConfig{})
		require.NoError(t, reg1.Register(&simpleTool{name: "tool1"}))

		composite := NewCompositeRegistry(reg1)

		_, err := composite.Execute(context.Background(), &llm.ToolRequest{
			ID:        "test-3",
			Name:      "nonexistent",
			Arguments: json.RawMessage(`{}`),
		})

		assert.ErrorIs(t, err, ErrToolNotFound)
	})

	t.Run("duplicate tool name - first registry wins", func(t *testing.T) {
		t.Parallel()

		reg1 := NewRegistry(RegistryConfig{})
		require.NoError(t, reg1.Register(&simpleTool{name: "shared", result: "from-reg1"}))

		reg2 := NewRegistry(RegistryConfig{})
		require.NoError(t, reg2.Register(&simpleTool{name: "shared", result: "from-reg2"}))

		composite := NewCompositeRegistry(reg1, reg2)

		resp, err := composite.Execute(context.Background(), &llm.ToolRequest{
			ID:        "test-4",
			Name:      "shared",
			Arguments: json.RawMessage(`{}`),
		})

		require.NoError(t, err)
		assert.Contains(t, string(resp.Result), "from-reg1", "first registry should execute")
	})
}

func TestCompositeRegistry_ExecuteAll(t *testing.T) {
	t.Parallel()

	t.Run("execute multiple tools from different registries", func(t *testing.T) {
		t.Parallel()

		reg1 := NewRegistry(RegistryConfig{})
		require.NoError(t, reg1.Register(&simpleTool{name: "tool1", result: "result1"}))

		reg2 := NewRegistry(RegistryConfig{})
		require.NoError(t, reg2.Register(&simpleTool{name: "tool2", result: "result2"}))

		composite := NewCompositeRegistry(reg1, reg2)

		reqs := []*llm.ToolRequest{
			{ID: "req-1", Name: "tool1", Arguments: json.RawMessage(`{}`)},
			{ID: "req-2", Name: "tool2", Arguments: json.RawMessage(`{}`)},
		}

		responses := composite.ExecuteAll(context.Background(), reqs)

		require.Len(t, responses, 2)
		assert.Equal(t, "req-1", responses[0].ID)
		assert.Contains(t, string(responses[0].Result), "result1")
		assert.Equal(t, "req-2", responses[1].ID)
		assert.Contains(t, string(responses[1].Result), "result2")
	})

	t.Run("execute with nonexistent tool returns error response", func(t *testing.T) {
		t.Parallel()

		reg1 := NewRegistry(RegistryConfig{})
		require.NoError(t, reg1.Register(&simpleTool{name: "tool1", result: "result1"}))

		composite := NewCompositeRegistry(reg1)

		reqs := []*llm.ToolRequest{
			{ID: "req-1", Name: "tool1", Arguments: json.RawMessage(`{}`)},
			{ID: "req-2", Name: "nonexistent", Arguments: json.RawMessage(`{}`)},
		}

		responses := composite.ExecuteAll(context.Background(), reqs)

		require.Len(t, responses, 2)
		assert.Equal(t, "req-1", responses[0].ID)
		assert.Empty(t, responses[0].Error)

		assert.Equal(t, "req-2", responses[1].ID)
		assert.NotEmpty(t, responses[1].Error)
	})
}

func TestCompositeRegistry_RegisterUnregisterNotSupported(t *testing.T) {
	t.Parallel()

	t.Run("register not supported", func(t *testing.T) {
		t.Parallel()

		composite := NewCompositeRegistry()
		err := composite.Register(&simpleTool{name: "tool1"})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "cannot register tools directly in composite registry")
	})

	t.Run("unregister not supported", func(t *testing.T) {
		t.Parallel()

		composite := NewCompositeRegistry()
		err := composite.Unregister("tool1")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "cannot unregister tools from composite registry")
	})
}

func TestCompositeRegistry_MCPUseCase(t *testing.T) {
	t.Parallel()

	// Simulate MCP registry with tools automatically registered
	mcpRegistry := NewRegistry(RegistryConfig{})
	require.NoError(t, mcpRegistry.Register(&simpleTool{name: "mcp_tool", result: "from-mcp"}))

	// Agent 1 has MCP tools + custom tools
	agent1Custom := NewRegistry(RegistryConfig{})
	require.NoError(t, agent1Custom.Register(&simpleTool{name: "agent1_custom", result: "agent1-result"}))
	agent1Registry := NewCompositeRegistry(mcpRegistry, agent1Custom)

	// Agent 2 has same MCP tools + different custom tools
	agent2Custom := NewRegistry(RegistryConfig{})
	require.NoError(t, agent2Custom.Register(&simpleTool{name: "agent2_custom", result: "agent2-result"}))
	agent2Registry := NewCompositeRegistry(mcpRegistry, agent2Custom)

	// Both agents can access MCP tools
	t.Run("agent1 can access MCP tool", func(t *testing.T) {
		resp, err := agent1Registry.Execute(context.Background(), &llm.ToolRequest{
			ID: "test-1", Name: "mcp_tool", Arguments: json.RawMessage(`{}`),
		})
		require.NoError(t, err)
		assert.Contains(t, string(resp.Result), "from-mcp")
	})

	t.Run("agent2 can access MCP tool", func(t *testing.T) {
		resp, err := agent2Registry.Execute(context.Background(), &llm.ToolRequest{
			ID: "test-2", Name: "mcp_tool", Arguments: json.RawMessage(`{}`),
		})
		require.NoError(t, err)
		assert.Contains(t, string(resp.Result), "from-mcp")
	})

	// Each agent has its own custom tools
	t.Run("agent1 has its custom tool", func(t *testing.T) {
		resp, err := agent1Registry.Execute(context.Background(), &llm.ToolRequest{
			ID: "test-3", Name: "agent1_custom", Arguments: json.RawMessage(`{}`),
		})
		require.NoError(t, err)
		assert.Contains(t, string(resp.Result), "agent1-result")
	})

	t.Run("agent2 has its custom tool", func(t *testing.T) {
		resp, err := agent2Registry.Execute(context.Background(), &llm.ToolRequest{
			ID: "test-4", Name: "agent2_custom", Arguments: json.RawMessage(`{}`),
		})
		require.NoError(t, err)
		assert.Contains(t, string(resp.Result), "agent2-result")
	})

	// Agents don't see each other's custom tools
	t.Run("agent1 cannot access agent2 custom tool", func(t *testing.T) {
		_, err := agent1Registry.Get("agent2_custom")
		assert.ErrorIs(t, err, ErrToolNotFound)
	})

	t.Run("agent2 cannot access agent1 custom tool", func(t *testing.T) {
		_, err := agent2Registry.Get("agent1_custom")
		assert.ErrorIs(t, err, ErrToolNotFound)
	})

	// Verify tool lists
	t.Run("agent1 lists MCP + custom tools", func(t *testing.T) {
		defs := agent1Registry.List()
		require.Len(t, defs, 2)
		names := []string{defs[0].Name, defs[1].Name}
		assert.Contains(t, names, "mcp_tool")
		assert.Contains(t, names, "agent1_custom")
	})

	t.Run("agent2 lists MCP + custom tools", func(t *testing.T) {
		defs := agent2Registry.List()
		require.Len(t, defs, 2)
		names := []string{defs[0].Name, defs[1].Name}
		assert.Contains(t, names, "mcp_tool")
		assert.Contains(t, names, "agent2_custom")
	})
}
