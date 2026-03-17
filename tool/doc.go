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
