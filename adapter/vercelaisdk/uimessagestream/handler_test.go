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

package uimessagestream

import (
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWriteChunk_DoesNotHTMLEscape(t *testing.T) {
	t.Parallel()

	rec := httptest.NewRecorder()
	ew := NewEventWriter(rec)

	require.NoError(t, ew.WriteChunk(Chunk{"type": "text-delta", "id": "text-0", "delta": "if a < b && c > d"}))

	body := rec.Body.String()
	assert.Contains(t, body, `"delta":"if a < b && c > d"`, "markup must be emitted verbatim, matching JSON.stringify")
	assert.NotContains(t, body, "\\u003c", "< must not be escaped")
	assert.NotContains(t, body, "\\u003e", "> must not be escaped")
	assert.NotContains(t, body, "\\u0026", "& must not be escaped")
}
