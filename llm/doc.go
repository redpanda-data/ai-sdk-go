// Package llm provides a unified interface for interacting with Large Language Models
// from different providers such as OpenAI, Anthropic, Google, and others.
//
// The central abstraction is the Model interface, which provides both streaming
// and non-streaming text generation with support for tool calling, structured
// output, and multimedia inputs.
//
// All provider implementations return the same unified types, ensuring that
// application code remains unchanged when switching between different LLM services.
// This enables easy testing, comparison, and migration between providers.
//
// Streaming follows Go idioms similar to sql.Rows and bufio.Scanner through the
// EventStream interface. All events use discriminated unions for compile-time
// type safety without exposing any interface{} types in the public API.
//
// The Part system allows extensible content representation that can grow from
// simple text to complex multimedia and tool interactions without breaking changes.
package llm
