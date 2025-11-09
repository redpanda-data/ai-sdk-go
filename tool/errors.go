package tool

import "errors"

// Sentinel errors for tool registry operations.
// These errors provide structured error handling and enable testing.
var (
	// ErrToolNil indicates that a nil tool was provided to Register.
	ErrToolNil = errors.New("tool cannot be nil")

	// ErrToolNameEmpty indicates that the tool has an empty name.
	ErrToolNameEmpty = errors.New("tool name cannot be empty")

	// ErrToolAlreadyRegistered indicates that a tool with the same name is already registered.
	ErrToolAlreadyRegistered = errors.New("tool already registered")

	// ErrToolNotFound indicates that the requested tool is not registered.
	ErrToolNotFound = errors.New("tool not found")

	// ErrToolRequestNil indicates that a nil tool request was provided to Execute.
	ErrToolRequestNil = errors.New("tool request cannot be nil")

	// ErrToolExecutionTimeout indicates that tool execution exceeded the configured timeout.
	ErrToolExecutionTimeout = errors.New("tool execution timeout")

	// ErrToolResponseTooLarge indicates that the tool response exceeds the configured size limit.
	ErrToolResponseTooLarge = errors.New("tool response too large")

	// ErrInvalidToolConfig indicates that the tool configuration is invalid.
	ErrInvalidToolConfig = errors.New("invalid tool configuration")
)
