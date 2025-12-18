package tool

import (
	"context"
	"errors"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// CompositeRegistry aggregates multiple registries and presents them as a single registry.
// Tools from all child registries are accessible through this composite view.
//
// Use cases:
// - Share MCP client tools across multiple agents without duplicate connections
// - Combine tools from different sources (MCP servers, built-ins, custom)
// - Organize tools by category or domain while presenting unified interface
//
// Example:
//
//	mcpRegistry := tool.NewRegistry(tool.RegistryConfig{})
//	// MCP client registers tools automatically
//	mcp.NewClient("server1", transport, mcp.WithRegistry(mcpRegistry))
//
//	// Create agent-specific registries that include MCP tools
//	agent1Registry := tool.NewCompositeRegistry(mcpRegistry, agent1CustomRegistry)
//	agent2Registry := tool.NewCompositeRegistry(mcpRegistry, agent2CustomRegistry)
//
//	// Both agents share same MCP client/tools, but have separate custom tools
type CompositeRegistry struct {
	registries []Registry
}

// NewCompositeRegistry creates a registry that aggregates multiple child registries.
// Tools are looked up in order - first registry with matching tool name wins.
// Tool lists are merged from all registries.
func NewCompositeRegistry(registries ...Registry) *CompositeRegistry {
	return &CompositeRegistry{
		registries: registries,
	}
}

// Register is not supported on composite registries - register in child registries instead.
func (c *CompositeRegistry) Register(_ Tool, _ ...Option) error {
	return errors.New("cannot register tools directly in composite registry: register in child registries instead")
}

// Unregister is not supported on composite registries - unregister from child registries instead.
func (c *CompositeRegistry) Unregister(_ string) error {
	return errors.New("cannot unregister tools from composite registry: unregister from child registries instead")
}

// List returns merged tool definitions from all child registries.
// If multiple registries have tools with the same name, only the first one is included.
func (c *CompositeRegistry) List() []llm.ToolDefinition {
	seen := make(map[string]bool)
	var result []llm.ToolDefinition

	for _, registry := range c.registries {
		for _, def := range registry.List() {
			if !seen[def.Name] {
				seen[def.Name] = true
				result = append(result, def)
			}
		}
	}

	return result
}

// Get retrieves a tool by name from the first registry that has it.
// Returns ErrToolNotFound if no child registry has this tool.
func (c *CompositeRegistry) Get(name string) (Tool, error) {
	for _, registry := range c.registries {
		t, err := registry.Get(name)
		if err == nil {
			return t, nil
		}
		// Continue to next registry if not found
		if errors.Is(err, ErrToolNotFound) {
			continue
		}
		// Propagate other errors
		return nil, err
	}

	return nil, ErrToolNotFound
}

// Execute runs a tool from the first registry that has it.
func (c *CompositeRegistry) Execute(ctx context.Context, req *llm.ToolRequest) (*llm.ToolResponse, error) {
	for _, registry := range c.registries {
		_, err := registry.Get(req.Name)
		if err == nil {
			return registry.Execute(ctx, req)
		}
		// Continue to next registry if not found
		if errors.Is(err, ErrToolNotFound) {
			continue
		}
		// Propagate other errors
		return nil, err
	}

	return nil, ErrToolNotFound
}

// ExecuteAll runs multiple tool requests concurrently, routing each to the appropriate child registry.
// Respects batch options like WithMaxConcurrency() for controlled concurrency.
//
// Like the normal registry, this uses the best-effort pattern: all errors are encoded in
// ToolResponse.Error fields, and len(reqs) responses are always returned in the same order.
func (c *CompositeRegistry) ExecuteAll(ctx context.Context, reqs []*llm.ToolRequest, opts ...BatchOption) []*llm.ToolResponse {
	// Use the shared concurrent execution helper with our Execute method
	// which routes each request to the appropriate child registry
	return executeAllConcurrent(ctx, reqs, opts, func(ctx context.Context, req *llm.ToolRequest) (*llm.ToolResponse, error) {
		if req == nil {
			return &llm.ToolResponse{Error: ErrToolRequestNil.Error()}, ErrToolRequestNil
		}

		resp, err := c.Execute(ctx, req)
		if resp == nil {
			resp = &llm.ToolResponse{ID: req.ID, Name: req.Name}
		}

		if err != nil {
			resp.Error = err.Error()
		}

		return resp, err
	})
}
