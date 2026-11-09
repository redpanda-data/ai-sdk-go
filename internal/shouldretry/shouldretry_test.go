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

package shouldretry

import (
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestForbidden(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		value string
		want  bool
	}{
		{name: "false", value: "false", want: true},
		{name: "False with spaces around it", value: " False ", want: true},
		{name: "true", value: "true", want: false},
		{name: "no header", value: "", want: false},
		{name: "a value other than true or false", value: "0", want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			h := http.Header{}
			if tt.value != "" {
				h.Set(Header, tt.value)
			}

			assert.Equal(t, tt.want, Forbidden(h))
		})
	}
}

func TestForbiddenBy_NilResponse(t *testing.T) {
	t.Parallel()

	assert.False(t, ForbiddenBy(nil))
}

func TestApply(t *testing.T) {
	t.Parallel()

	newErr := func() error {
		return fmt.Errorf("%w: %w", llm.ErrAPICall, &llm.ProviderError{Base: llm.ErrServerError, Retryable: true})
	}

	assert.False(t, llm.IsRetryable(Apply(newErr(), true)), "Apply should clear Retryable on a ProviderError inside a wrapped error")
	assert.True(t, llm.IsRetryable(Apply(newErr(), false)), "Apply with forbidden false should leave Retryable true")

	plain := errors.New("not a provider error")
	assert.Equal(t, plain, Apply(plain, true))
}

func TestTransport_AnyStatusSetsTheFlag(t *testing.T) {
	t.Parallel()

	for _, status := range []int{http.StatusOK, http.StatusServiceUnavailable} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			t.Parallel()

			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set(Header, "false")
				w.WriteHeader(status)
			}))
			t.Cleanup(srv.Close)

			ctx, flag := Watch(t.Context())
			req, err := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL, nil)
			require.NoError(t, err)

			resp, err := (&http.Client{Transport: Transport{Base: srv.Client().Transport}}).Do(req)
			require.NoError(t, err)
			require.NoError(t, resp.Body.Close())

			assert.True(t, flag.Load(), "a %d with x-should-retry: false should set the flag", status)
		})
	}
}
