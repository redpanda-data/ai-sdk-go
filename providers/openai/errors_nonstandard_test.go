package openai

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestClassifyHTTPError_NonStandardErrorBody(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		status      int
		contentType string
		body        string
	}{
		{
			name:        "HTML 404 from wrong base URL",
			status:      http.StatusNotFound,
			contentType: "text/html",
			body:        `<html><body><h1>404 Not Found</h1></body></html>`,
		},
		{
			name:        "plain text error",
			status:      http.StatusBadGateway,
			contentType: "text/plain",
			body:        `upstream connect error or disconnect/reset before headers`,
		},
		{
			name:        "non-OpenAI JSON error",
			status:      http.StatusNotFound,
			contentType: "application/json",
			body:        `{"status":"not_found","detail":"no route matched"}`,
		},
		{
			name:        "empty body",
			status:      http.StatusBadGateway,
			contentType: "application/json",
			body:        ``,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", tt.contentType)
				w.WriteHeader(tt.status)
				fmt.Fprint(w, tt.body)
			}))
			defer srv.Close()

			provider, err := NewProvider("sk-test", WithBaseURL(srv.URL))
			require.NoError(t, err)

			model, err := provider.NewModel("gpt-4o")
			require.NoError(t, err)

			_, err = model.Generate(context.Background(), &llm.Request{
				Messages: []llm.Message{{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("hello")}}},
			})
			require.Error(t, err)

			var pe *llm.ProviderError
			require.ErrorAs(t, err, &pe)
			assert.NotEmpty(t, pe.Message,
				"ProviderError.Message must not be empty for non-standard error responses")
			assert.Contains(t, pe.Error(), fmt.Sprintf("%d", tt.status),
				"error string should contain the HTTP status code")
		})
	}
}

func TestClassifyHTTPError_NonStandardErrorBody_Stream(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprint(w, `<html><body><h1>404 Not Found</h1></body></html>`)
	}))
	defer srv.Close()

	provider, err := NewProvider("sk-test", WithBaseURL(srv.URL))
	require.NoError(t, err)

	model, err := provider.NewModel("gpt-4o")
	require.NoError(t, err)

	var streamErr error
	for _, err := range model.GenerateEvents(context.Background(), &llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("hello")}}},
	}) {
		if err != nil {
			streamErr = err
		}
	}
	require.Error(t, streamErr)

	var pe *llm.ProviderError
	require.ErrorAs(t, streamErr, &pe)
	assert.NotEmpty(t, pe.Message,
		"ProviderError.Message must not be empty for non-standard streaming error responses")
}
