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

package llmagent

import (
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

func TestToolResponse(t *testing.T) {
	t.Parallel()

	req := &llm.ToolRequestPart{ID: "call_1", Name: "search"}
	payload := json.RawMessage(`[{"type":"text","text":"HTTP 400"}]`)

	t.Run("result error keeps the tool payload", func(t *testing.T) {
		t.Parallel()

		err := fmt.Errorf("tool %q failed: %w", req.Name, &tool.ResultError{Result: payload})
		resp := toolResponse(req, nil, err)
		assert.True(t, resp.IsError)
		assert.JSONEq(t, string(payload), string(resp.Result))
	})

	t.Run("other errors are wrapped", func(t *testing.T) {
		t.Parallel()

		resp := toolResponse(req, nil, errors.New("connection closed"))
		assert.True(t, resp.IsError)
		assert.JSONEq(t, `{"error":"connection closed"}`, string(resp.Result))
	})
}
