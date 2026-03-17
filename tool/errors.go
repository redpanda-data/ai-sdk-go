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
