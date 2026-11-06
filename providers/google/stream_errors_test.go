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

package google

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestUnwrapErrorEvent(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		line string
		want string
	}{
		{
			name: "an error event loses its data: prefix",
			line: "data: {\"error\":{\"code\":503,\"status\":\"UNAVAILABLE\"}}\n",
			want: "{\"error\":{\"code\":503,\"status\":\"UNAVAILABLE\"}}\n",
		},
		{
			name: "a chunk whose text mentions an error is unchanged",
			line: "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"the \\\"error\\\" key\"}]}}]}\n",
			want: "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"the \\\"error\\\" key\"}]}}]}\n",
		},
		{
			name: "a null error is unchanged",
			line: "data: {\"error\":null}\n",
			want: "data: {\"error\":null}\n",
		},
		{
			name: "a line that isn't data is unchanged",
			line: "\n",
			want: "\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			assert.Equal(t, tt.want, string(unwrapErrorEvent([]byte(tt.line))))
		})
	}
}
