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

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/google/jsonschema-go/jsonschema"
)

// Func wraps a typed function into a Tool. The function receives a
// decoded In value and returns a Result[Out] (typically constructed via
// Done, Pending, or NeedInput). The registry handles JSON marshaling,
// schema inference (when Spec.InputSchema is empty), and Await
// validation.
//
// Use tool.Must to discard the error in package-level declarations:
//
//	var addTool = tool.Must(tool.Func(
//	    tool.Spec{Name: "add", Description: "Add two integers."},
//	    func(ctx context.Context, in AddInput) (tool.Result[AddOutput], error) {
//	        return tool.Done(AddOutput{Sum: in.A + in.B}), nil
//	    },
//	))
//
// Spec.Name is required. Other Spec fields are optional; InputSchema is
// inferred from In via reflection when omitted.
func Func[In, Out any](spec Spec, fn func(ctx context.Context, in In) (Result[Out], error)) (Tool, error) {
	if spec.Name == "" {
		return nil, ErrToolNameEmpty
	}

	if fn == nil {
		return nil, errors.New("tool: Func body cannot be nil")
	}

	schema := spec.InputSchema
	if len(schema) == 0 {
		inferred, err := inferInputSchema[In]()
		if err != nil {
			return nil, fmt.Errorf("tool %q: infer input schema: %w", spec.Name, err)
		}

		schema = inferred
	}

	// Cache the schema bytes on the Spec so Definition() and any
	// subsequent introspection see the same value.
	spec.InputSchema = schema

	return &funcTool[In, Out]{spec: spec, fn: fn}, nil
}

// Must panics if err is non-nil, otherwise returns t. Intended for
// package-level Tool declarations where the inputs are static and a
// panic at init is the right failure mode.
func Must(t Tool, err error) Tool {
	if err != nil {
		panic(fmt.Errorf("tool.Must: %w", err)) //nolint:forbidigo // init-time programmer error
	}

	return t
}

// funcTool is the concrete Tool implementation produced by Func.
type funcTool[In, Out any] struct {
	spec Spec
	fn   func(ctx context.Context, in In) (Result[Out], error)
}

func (t *funcTool[In, Out]) Name() string                 { return t.spec.Name }
func (t *funcTool[In, Out]) Description() string          { return t.spec.Description }
func (t *funcTool[In, Out]) InputSchema() json.RawMessage { return t.spec.InputSchema }
func (t *funcTool[In, Out]) ToolSpec() Spec               { return t.spec }

// Execute decodes call.Args into In, runs the user function, and
// marshals the typed result.
//
// Re-entry (call.Resume != nil) never re-runs fn — re-running would
// repeat the function's side effects with the original arguments. The
// caller's submission resolves the call directly: Resume.Error becomes a
// tool error, otherwise Resume.Result is the final output. Tools that
// need custom re-entry logic (post-processing the resume payload,
// chaining another pause) should implement Tool directly.
func (t *funcTool[In, Out]) Execute(ctx context.Context, call Call) (Execution, error) {
	if call.Resume != nil {
		if call.Resume.Error != "" {
			return Execution{}, fmt.Errorf("tool %q: %s", t.spec.Name, call.Resume.Error)
		}

		return Execution{Output: call.Resume.Result}, nil
	}

	var in In

	args := call.Args
	if len(args) > 0 {
		// Empty {} input is allowed for tools with no fields.
		if err := json.Unmarshal(args, &in); err != nil {
			return Execution{}, fmt.Errorf("tool %q: decode arguments: %w", t.spec.Name, err)
		}
	}

	result, err := t.fn(ctx, in)
	if err != nil {
		return Execution{}, err
	}

	// The registry is the enforcement point for Await validity and
	// AsyncSpec consistency; this early check only exists to attribute a
	// malformed Await to the tool by name before it leaves the typed path.
	if err := result.Await.Validate(); err != nil {
		return Execution{}, fmt.Errorf("tool %q: %w", t.spec.Name, err)
	}

	output, err := json.Marshal(result.Value)
	if err != nil {
		return Execution{}, fmt.Errorf("tool %q: encode output: %w", t.spec.Name, err)
	}

	return Execution{
		Output:  output,
		Await:   result.Await,
		Actions: result.Actions,
	}, nil
}

// inferInputSchema returns a JSON Schema for the In type via the
// google/jsonschema-go reflector. Tools that need to override the
// inferred schema can set Spec.InputSchema explicitly.
func inferInputSchema[In any]() (json.RawMessage, error) {
	schema, err := jsonschema.For[In](nil)
	if err != nil {
		return nil, err
	}

	if schema == nil {
		return json.RawMessage(`{"type":"object"}`), nil
	}

	return json.Marshal(schema)
}
