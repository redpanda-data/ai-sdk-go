// Package tool provides a runtime for registering and executing functions
// that can be called by Large Language Models.
//
// Registry manages tool registration and execution. Create with NewRegistry
// and register tools using functional options:
//
//	registry := tool.NewRegistry(tool.RegistryConfig{})
//	registry.Register(myTool, tool.WithTimeout(30*time.Second))
//
// Tool is the basic interface - implement Definition and Execute.
//
// Tools integrate with LLMs via the llm package's wire protocol types
// (ToolDefinition, ToolRequest, ToolResponse). The registry provides
// tool definitions and executes tool requests.
//
// Basic usage:
//
//	registry.Register(myTool, tool.WithTimeout(1*time.Minute))
//	result, err := registry.Execute(ctx, toolRequest)
package tool
