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

package webfetch

import (
	"encoding/json"

	"github.com/redpanda-data/ai-sdk-go/tool"
)

// errorExecution returns an Execution whose Output is a structured
// `{"error":true,"message":...}` payload — preserving the legacy
// webfetch error shape that the model has been prompted on — but with a
// nil top-level error so the runtime still treats the call as
// successful. webfetch is documented to encode its own errors this way;
// the wrapped error stays out of band.
func errorExecution(err error) tool.Execution {
	payload := map[string]any{
		"error":   true,
		"message": err.Error(),
	}

	data, marshalErr := json.Marshal(payload)
	if marshalErr != nil {
		data = json.RawMessage(`{"error": true, "message": "internal error"}`)
	}

	return tool.Execution{Output: data}
}
