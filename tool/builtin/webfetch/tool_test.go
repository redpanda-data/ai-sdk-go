package webfetch

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// testOptions returns common test options that disable security restrictions for testing.
func testOptions() []Option {
	return []Option{
		WithDenyPrivateIPs(false),                     // Allow localhost for testing
		WithAllowedSchemes([]string{"https", "http"}), // Allow both schemes
		WithAllowedPorts(nil),                         // Allow all ports in tests
	}
}

func TestWebFetch_EndToEnd(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name         string
		setupServer  func() *httptest.Server
		args         string
		toolOptions  []Option
		validateResp func(t *testing.T, resp map[string]any)
	}{
		{
			name: "successful GET request with HTML content",
			setupServer: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					assert.Equal(t, "GET", r.Method)
					assert.Contains(t, r.Header.Get("User-Agent"), "Redpanda AI-Agent-SDK")

					w.Header().Set("Content-Type", "text/html; charset=utf-8")
					w.WriteHeader(http.StatusOK)
					_, _ = w.Write([]byte("<html><body><h1>Test Page</h1><p>Hello World!</p></body></html>"))
				}))
			},
			args:        `{"url": "%s", "method": "GET"}`,
			toolOptions: testOptions(),
			validateResp: func(t *testing.T, resp map[string]any) {
				t.Helper()
				// Check if there's an error first
				if errVal, hasError := resp["error"]; hasError && errVal != nil {
					errBool, ok := errVal.(bool)
					require.True(t, ok, "error should be a bool")
					require.False(t, errBool, "Unexpected error: %v", resp["message"])
				}

				require.Contains(t, resp, "status_code")
				statusCode, ok := resp["status_code"].(float64)
				require.True(t, ok, "status_code should be a float64")
				assert.Equal(t, 200, int(statusCode))
				assert.Equal(t, "200 OK", resp["status"])
				assert.Equal(t, "text/html", resp["media_type"])
				assert.Equal(t, "utf-8", resp["encoding"])

				// Verify fencing is applied by default
				body, ok := resp["body"].(string)
				require.True(t, ok, "body should be a string")
				assert.True(t, strings.HasPrefix(body, "```untrusted_text\n"), "body should start with fence marker")
				assert.Contains(t, body, "Test Page")
				assert.Contains(t, body, "Hello World!")
				assert.True(t, strings.HasSuffix(strings.TrimSpace(body), "```"), "body should end with fence marker")

				truncated, ok := resp["truncated"].(bool)
				require.True(t, ok, "truncated should be a bool")
				assert.False(t, truncated)

				redirected, ok := resp["redirected"].(bool)
				require.True(t, ok, "redirected should be a bool")
				assert.False(t, redirected)

				assert.Contains(t, resp, "retrieved_at")
			},
		},
		{
			name: "HEAD request returns no body",
			setupServer: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					assert.Equal(t, "HEAD", r.Method)
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusOK)
				}))
			},
			args:        `{"url": "%s", "method": "HEAD"}`,
			toolOptions: testOptions(),
			validateResp: func(t *testing.T, resp map[string]any) {
				t.Helper()

				if errVal, hasError := resp["error"]; hasError && errVal != nil {
					errBool, ok := errVal.(bool)
					require.True(t, ok, "error should be a bool")
					require.False(t, errBool, "Unexpected error: %v", resp["message"])
				}

				require.Contains(t, resp, "status_code")
				statusCode, ok := resp["status_code"].(float64)
				require.True(t, ok, "status_code should be a float64")
				assert.Equal(t, 200, int(statusCode))
				assert.Equal(t, "application/json", resp["media_type"])
				assert.NotContains(t, resp, "body") // HEAD should not include body

				redirected, ok := resp["redirected"].(bool)
				require.True(t, ok, "redirected should be a bool")
				assert.False(t, redirected)
			},
		},
		{
			name: "handles redirects properly",
			setupServer: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.URL.Path == "/redirect" {
						http.Redirect(w, r, "/final", http.StatusFound)
						return
					}

					if r.URL.Path == "/final" {
						w.Header().Set("Content-Type", "text/plain")
						w.WriteHeader(http.StatusOK)
						_, _ = w.Write([]byte("Final destination"))

						return
					}

					w.WriteHeader(http.StatusNotFound)
				}))
			},
			args:        `{"url": "%s/redirect"}`,
			toolOptions: testOptions(),
			validateResp: func(t *testing.T, resp map[string]any) {
				t.Helper()

				if errVal, hasError := resp["error"]; hasError && errVal != nil {
					errBool, ok := errVal.(bool)
					require.True(t, ok, "error should be a bool")
					require.False(t, errBool, "Unexpected error: %v", resp["message"])
				}

				require.Contains(t, resp, "status_code")
				statusCode, ok := resp["status_code"].(float64)
				require.True(t, ok, "status_code should be a float64")
				assert.Equal(t, 200, int(statusCode))

				redirected, ok := resp["redirected"].(bool)
				require.True(t, ok, "redirected should be a bool")
				assert.True(t, redirected)

				assert.Contains(t, resp["final_url"], "/final")

				body, ok := resp["body"].(string)
				require.True(t, ok, "body should be a string")
				assert.Contains(t, body, "Final destination")
			},
		},
		{
			name: "converts HTML to markdown by default",
			setupServer: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.Header().Set("Content-Type", "text/html; charset=utf-8")
					w.WriteHeader(http.StatusOK)
					// Write complex HTML to test markdown conversion
					htmlContent := `<!DOCTYPE html>
<html>
<head>
	<title>Test Document</title>
	<script>console.log('should be removed');</script>
	<style>.test { color: red; }</style>
</head>
<body>
	<nav>Navigation menu</nav>
	<h1>Main Heading</h1>
	<h2>Sub Heading</h2>
	<p>This is a <strong>bold text</strong> and <em>italic text</em>.</p>
	<ul>
		<li>First item</li>
		<li>Second item</li>
	</ul>
	<blockquote>This is a quote</blockquote>
	<code>inline code</code>
	<pre><code>block code</code></pre>
	<a href="/relative-link">Relative link</a>
	<a href="https://example.com">Absolute link</a>
	<footer>Footer content</footer>
</body>
</html>`
					_, _ = w.Write([]byte(htmlContent))
				}))
			},
			args:        `{"url": "%s", "method": "GET"}`,
			toolOptions: testOptions(),
			validateResp: func(t *testing.T, resp map[string]any) {
				t.Helper()

				if errVal, hasError := resp["error"]; hasError && errVal != nil {
					errBool, ok := errVal.(bool)
					require.True(t, ok, "error should be a bool")
					require.False(t, errBool, "Unexpected error: %v", resp["message"])
				}

				require.Contains(t, resp, "status_code")
				statusCode, ok := resp["status_code"].(float64)
				require.True(t, ok, "status_code should be a float64")
				assert.Equal(t, 200, int(statusCode))
				assert.Equal(t, "text/html", resp["media_type"])

				// Check that markdown conversion happened
				require.Contains(t, resp, "converted_to_markdown")
				convertedToMarkdown, ok := resp["converted_to_markdown"].(bool)
				require.True(t, ok, "converted_to_markdown should be a bool")
				assert.True(t, convertedToMarkdown)

				body, ok := resp["body"].(string)
				require.True(t, ok, "body should be a string")
				// Verify markdown format
				assert.Contains(t, body, "# Main Heading")
				assert.Contains(t, body, "## Sub Heading")
				assert.Contains(t, body, "**bold text**")
				assert.Contains(t, body, "_italic text_")
				assert.Contains(t, body, "- First item")
				assert.Contains(t, body, "- Second item")
				assert.Contains(t, body, "> This is a quote")
				assert.Contains(t, body, "`inline code`")
				assert.Contains(t, body, "```")
				assert.Contains(t, body, "block code")
				assert.Contains(t, body, "[Absolute link](https://example.com)")

				// Verify unwanted elements are removed
				assert.NotContains(t, body, "script")
				assert.NotContains(t, body, "console.log")
				assert.NotContains(t, body, "style")
				assert.NotContains(t, body, "color: red")
				assert.NotContains(t, body, "Navigation menu")
				assert.NotContains(t, body, "Footer content")

				// Verify relative links are converted (may have URL encoding)
				assert.Contains(t, body, "[Relative link]")
				assert.Contains(t, body, "/relative-link")
			},
		},
		{
			name: "respects convert_to_markdown=false parameter",
			setupServer: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.Header().Set("Content-Type", "text/html; charset=utf-8")
					w.WriteHeader(http.StatusOK)
					_, _ = w.Write([]byte("<html><body><h1>Test Page</h1><p>Hello <strong>World!</strong></p></body></html>"))
				}))
			},
			args:        `{"url": "%s", "method": "GET", "convert_to_markdown": false}`,
			toolOptions: testOptions(),
			validateResp: func(t *testing.T, resp map[string]any) {
				t.Helper()

				if errVal, hasError := resp["error"]; hasError && errVal != nil {
					errBool, ok := errVal.(bool)
					require.True(t, ok, "error should be a bool")
					require.False(t, errBool, "Unexpected error: %v", resp["message"])
				}

				require.Contains(t, resp, "status_code")
				statusCode, ok := resp["status_code"].(float64)
				require.True(t, ok, "status_code should be a float64")
				assert.Equal(t, 200, int(statusCode))

				// Check that markdown conversion was disabled
				require.Contains(t, resp, "converted_to_markdown")
				convertedToMarkdown, ok := resp["converted_to_markdown"].(bool)
				require.True(t, ok, "converted_to_markdown should be a bool")
				assert.False(t, convertedToMarkdown)

				body, ok := resp["body"].(string)
				require.True(t, ok, "body should be a string")
				// Should contain original HTML
				assert.Contains(t, body, "<html>")
				assert.Contains(t, body, "<h1>Test Page</h1>")
				assert.Contains(t, body, "<strong>World!</strong>")
				assert.NotContains(t, body, "# Test Page") // No markdown conversion
			},
		},
		{
			name: "converts non-HTML content gracefully",
			setupServer: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusOK)
					_, _ = w.Write([]byte(`{"message": "Hello World", "status": "success"}`))
				}))
			},
			args:        `{"url": "%s", "method": "GET", "convert_to_markdown": true}`,
			toolOptions: testOptions(),
			validateResp: func(t *testing.T, resp map[string]any) {
				t.Helper()

				if errVal, hasError := resp["error"]; hasError && errVal != nil {
					errBool, ok := errVal.(bool)
					require.True(t, ok, "error should be a bool")
					require.False(t, errBool, "Unexpected error: %v", resp["message"])
				}

				require.Contains(t, resp, "status_code")
				statusCode, ok := resp["status_code"].(float64)
				require.True(t, ok, "status_code should be a float64")
				assert.Equal(t, 200, int(statusCode))
				assert.Equal(t, "application/json", resp["media_type"])

				// Check that markdown conversion was attempted and succeeded
				// (html-to-markdown gracefully handles non-HTML content)
				require.Contains(t, resp, "converted_to_markdown")
				convertedToMarkdown, ok := resp["converted_to_markdown"].(bool)
				require.True(t, ok, "converted_to_markdown should be a bool")
				assert.True(t, convertedToMarkdown)

				body, ok := resp["body"].(string)
				require.True(t, ok, "body should be a string")
				// JSON should be preserved as-is since it's not HTML
				assert.Contains(t, body, `"message": "Hello World"`)
				assert.Contains(t, body, `"status": "success"`)
			},
		},
		{
			name: "WithFencing(false) disables fencing",
			setupServer: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.Header().Set("Content-Type", "text/html; charset=utf-8")
					w.WriteHeader(http.StatusOK)
					_, _ = w.Write([]byte("<html><body><h1>Test Page</h1><p>Content without fencing</p></body></html>"))
				}))
			},
			args:        `{"url": "%s", "method": "GET"}`,
			toolOptions: append(testOptions(), WithFencing(false)),
			validateResp: func(t *testing.T, resp map[string]any) {
				t.Helper()

				if errVal, hasError := resp["error"]; hasError && errVal != nil {
					errBool, ok := errVal.(bool)
					require.True(t, ok, "error should be a bool")
					require.False(t, errBool, "Unexpected error: %v", resp["message"])
				}

				require.Contains(t, resp, "status_code")
				statusCode, ok := resp["status_code"].(float64)
				require.True(t, ok, "status_code should be a float64")
				assert.Equal(t, 200, int(statusCode))

				// Verify body is NOT fenced
				body, ok := resp["body"].(string)
				require.True(t, ok, "body should be a string")
				assert.NotContains(t, body, "```untrusted_text", "body should NOT be fenced when fencing is disabled")
				assert.Contains(t, body, "# Test Page")
				assert.Contains(t, body, "Content without fencing")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			server := tt.setupServer()
			defer server.Close()

			// Create tool with test-friendly options
			tool := New(tt.toolOptions...)
			ctx := context.Background()

			// Format URL into args
			args := fmt.Sprintf(tt.args, server.URL)

			result, err := tool.Execute(ctx, json.RawMessage(args))
			require.NoError(t, err)

			var response map[string]any
			require.NoError(t, json.Unmarshal(result, &response))

			tt.validateResp(t, response)
		})
	}
}

func TestWebFetch_SecurityValidation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		args        string
		toolOptions []Option
		expectError string
	}{
		{
			name:        "blocks HTTP scheme by default",
			args:        `{"url": "http://example.com"}`,
			expectError: "scheme",
		},
		{
			name:        "blocks non-443 ports by default",
			args:        `{"url": "https://example.com:8080"}`,
			expectError: "port",
		},
		{
			name:        "blocks unsupported method",
			args:        `{"url": "https://example.com", "method": "POST"}`,
			expectError: "unsupported method",
		},
		{
			name:        "blocks private IPs when enabled",
			args:        `{"url": "https://127.0.0.1"}`,
			expectError: "private",
		},
		{
			name:        "blocks credentials in URL",
			args:        `{"url": "https://user:pass@example.com"}`,
			expectError: "credentials",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			tool := New(tt.toolOptions...)
			ctx := context.Background()

			result, err := tool.Execute(ctx, json.RawMessage(tt.args))
			require.NoError(t, err)

			var response map[string]any
			require.NoError(t, json.Unmarshal(result, &response))

			// All cases should be errors
			require.Contains(t, response, "error")
			require.NotNil(t, response["error"])
			errBool, ok := response["error"].(bool)
			require.True(t, ok, "error should be a bool")
			assert.True(t, errBool, "Should have error")

			if msg, ok := response["message"]; ok && msg != nil {
				msgStr, ok := msg.(string)
				require.True(t, ok, "message should be a string")
				assert.Contains(t, strings.ToLower(msgStr), tt.expectError)
			}
		})
	}
}

func TestSSRFProtection(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		addr    string
		blocked bool
	}{
		// Public addresses (allowed)
		{"Google DNS", "8.8.8.8", false},
		{"Cloudflare DNS", "1.1.1.1", false},

		// Private ranges (blocked)
		{"RFC1918 10.x", "10.0.0.1", true},
		{"RFC1918 172.16.x", "172.16.0.1", true},
		{"RFC1918 192.168.x", "192.168.1.1", true},
		{"Localhost IPv4", "127.0.0.1", true},
		{"Localhost IPv6", "::1", true},
		{"Link-local", "169.254.1.1", true},
		{"Carrier-Grade NAT", "100.64.0.1", true},
		{"Multicast", "224.0.0.1", true},
		{"Reserved", "240.0.0.1", true},

		// New IPv6 special-purpose ranges
		{"IPv6 Documentation", "2001:db8::1", true},
		{"IPv6 ORCHID", "2001:10::1", true},
		{"IPv6 Site-Local", "fec0::1", true},
		{"IPv6 NAT64", "64:ff9b::1", true},

		// New IPv4 special-purpose ranges
		{"Documentation TEST-NET-1", "192.0.2.1", true},
		{"Documentation TEST-NET-2", "198.51.100.1", true},
		{"Benchmarking", "198.18.0.1", true},
		{"6to4 Relay", "192.88.99.1", true},

		// IPv4-mapped IPv6
		{"IPv4-mapped private", "::ffff:192.168.1.1", true},
		{"IPv4-mapped public", "::ffff:8.8.8.8", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			addr, err := netip.ParseAddr(tt.addr)
			require.NoError(t, err)

			result := isPrivateOrReserved(addr)
			assert.Equal(t, tt.blocked, result,
				"Address %s: expected blocked=%v, got %v", tt.addr, tt.blocked, result)
		})
	}
}

func TestSecurityHardening(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		args        string
		expectError string
	}{
		{
			name:        "blocks URLs with zone identifiers",
			args:        `{"url": "https://[fe80::1%25eth0]:443"}`,
			expectError: "zone identifiers not allowed",
		},
		{
			name:        "blocks extremely long URLs",
			args:        fmt.Sprintf(`{"url": "https://example.com/%s"}`, strings.Repeat("a", 9000)),
			expectError: "url too long",
		},
		{
			name:        "blocks file:// scheme",
			args:        `{"url": "file:///etc/passwd"}`,
			expectError: "scheme",
		},
		{
			name:        "blocks ftp:// scheme",
			args:        `{"url": "ftp://example.com/file.txt"}`,
			expectError: "scheme",
		},
		{
			name:        "blocks gopher:// scheme",
			args:        `{"url": "gopher://example.com/"}`,
			expectError: "scheme",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			tool := New() // Use default secure config
			ctx := context.Background()

			result, err := tool.Execute(ctx, json.RawMessage(tt.args))
			require.NoError(t, err)

			var response map[string]any
			require.NoError(t, json.Unmarshal(result, &response))

			require.Contains(t, response, "error")
			require.NotNil(t, response["error"])
			errBool, ok := response["error"].(bool)
			require.True(t, ok, "error should be a bool")
			assert.True(t, errBool)

			if msg, ok := response["message"]; ok && msg != nil {
				msgStr, ok := msg.(string)
				require.True(t, ok, "message should be a string")
				assert.Contains(t, strings.ToLower(msgStr), tt.expectError)
			}
		})
	}
}

func TestFencing(t *testing.T) { //nolint:paralleltest // shared httptest server across subtests
	// Create test server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("<html><body>Test content</body></html>"))
	}))
	defer server.Close()

	t.Run("fencing is enabled by default", func(t *testing.T) { //nolint:paralleltest // shared httptest server across subtests
		tool := New(testOptions()...)
		params := NewParameters(server.URL).WithMethod("GET")
		result, err := tool.Execute(context.Background(), params.MustToJSONRawMessage())
		require.NoError(t, err)

		var response map[string]any

		err = json.Unmarshal(result, &response)
		require.NoError(t, err)

		body, ok := response["body"].(string)
		require.True(t, ok, "body should be a string")
		assert.True(t, strings.HasPrefix(body, "```untrusted_text\n"), "body should start with fence marker")
		assert.Contains(t, body, "Test content", "body should contain actual content")
		assert.True(t, strings.HasSuffix(strings.TrimSpace(body), "```"), "body should end with fence end marker")
	})

	t.Run("WithFencing(false) disables fencing", func(t *testing.T) { //nolint:paralleltest // shared httptest server across subtests
		opts := append(testOptions(), WithFencing(false))
		tool := New(opts...)
		params := NewParameters(server.URL).WithMethod("GET")
		result, err := tool.Execute(context.Background(), params.MustToJSONRawMessage())
		require.NoError(t, err)

		var response map[string]any

		err = json.Unmarshal(result, &response)
		require.NoError(t, err)

		body, ok := response["body"].(string)
		require.True(t, ok, "body should be a string")
		assert.NotContains(t, body, "```untrusted_text", "body should NOT contain fence markers when disabled")
		assert.Contains(t, body, "Test content", "body should contain actual content")
	})
}
