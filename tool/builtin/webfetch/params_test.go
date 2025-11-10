package webfetch

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParameters(t *testing.T) {
	t.Parallel()

	t.Run("NewParameters creates default parameters", func(t *testing.T) {
		t.Parallel()

		params := NewParameters("https://example.com")

		assert.Equal(t, "https://example.com", params.URL)
		assert.Equal(t, http.MethodGet, params.Method)
		assert.Nil(t, params.ConvertToMarkdown)
	})

	t.Run("WithMethod sets method", func(t *testing.T) {
		t.Parallel()

		params := NewParameters("https://example.com").WithMethod(http.MethodHead)

		assert.Equal(t, "https://example.com", params.URL)
		assert.Equal(t, http.MethodHead, params.Method)
		assert.Nil(t, params.ConvertToMarkdown)
	})

	t.Run("WithMarkdownConversion sets conversion flag", func(t *testing.T) {
		t.Parallel()

		params := NewParameters("https://example.com").WithMarkdownConversion(true)

		assert.Equal(t, "https://example.com", params.URL)
		assert.Equal(t, http.MethodGet, params.Method)
		require.NotNil(t, params.ConvertToMarkdown)
		assert.True(t, *params.ConvertToMarkdown)
	})

	t.Run("chaining methods works correctly", func(t *testing.T) {
		t.Parallel()

		params := NewParameters("https://example.com").
			WithMethod(http.MethodHead).
			WithMarkdownConversion(false)

		assert.Equal(t, "https://example.com", params.URL)
		assert.Equal(t, http.MethodHead, params.Method)
		require.NotNil(t, params.ConvertToMarkdown)
		assert.False(t, *params.ConvertToMarkdown)
	})

	t.Run("ToJSONRawMessage serializes correctly", func(t *testing.T) {
		t.Parallel()

		params := NewParameters("https://example.com").
			WithMethod(http.MethodGet).
			WithMarkdownConversion(true)

		rawMsg, err := params.ToJSONRawMessage()
		require.NoError(t, err)

		// Deserialize and verify
		var result map[string]any

		err = json.Unmarshal(rawMsg, &result)
		require.NoError(t, err)

		assert.Equal(t, "https://example.com", result["url"])
		assert.Equal(t, "GET", result["method"])
		assert.Equal(t, true, result["convert_to_markdown"])
	})

	t.Run("MustToJSONRawMessage works correctly", func(t *testing.T) {
		t.Parallel()

		params := NewParameters("https://example.com")

		// Should not panic
		rawMsg := params.MustToJSONRawMessage()
		assert.NotNil(t, rawMsg)

		// Verify content
		var result map[string]any

		err := json.Unmarshal(rawMsg, &result)
		require.NoError(t, err)
		assert.Equal(t, "https://example.com", result["url"])
		assert.Equal(t, "GET", result["method"])
		// convert_to_markdown should not be present when nil
		_, hasConvert := result["convert_to_markdown"]
		assert.False(t, hasConvert)
	})

	t.Run("serialization omits empty optional fields", func(t *testing.T) {
		t.Parallel()

		// Only URL set, method uses default
		params := &Parameters{
			URL: "https://example.com",
		}

		rawMsg, err := params.ToJSONRawMessage()
		require.NoError(t, err)

		var result map[string]any

		err = json.Unmarshal(rawMsg, &result)
		require.NoError(t, err)

		assert.Equal(t, "https://example.com", result["url"])
		// method should not be present when empty (omitempty)
		_, hasMethod := result["method"]
		assert.False(t, hasMethod)
		// convert_to_markdown should not be present when nil
		_, hasConvert := result["convert_to_markdown"]
		assert.False(t, hasConvert)
	})

	t.Run("serialization includes explicit method when set", func(t *testing.T) {
		t.Parallel()

		params := &Parameters{
			URL:    "https://example.com",
			Method: http.MethodHead,
		}

		rawMsg, err := params.ToJSONRawMessage()
		require.NoError(t, err)

		var result map[string]any

		err = json.Unmarshal(rawMsg, &result)
		require.NoError(t, err)

		assert.Equal(t, "https://example.com", result["url"])
		assert.Equal(t, "HEAD", result["method"])
	})
}
