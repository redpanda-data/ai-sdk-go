package webfetch

import "fmt"

// FenceConfig configures how untrusted content is fenced/wrapped
// to protect against prompt injection attacks. The default configuration
// uses OpenAI's recommended untrusted_text block format.
//
// References:
//   - https://model-spec.openai.com/2025-02-12.html (Authority Hierarchy)
//   - https://cookbook.openai.com/articles/openai-harmony (Harmony Format)
type FenceConfig struct {
	// Start is the opening delimiter for untrusted content.
	// Default: "```untrusted_text"
	Start string

	// End is the closing delimiter for untrusted content.
	// Default: "```"
	End string
}

// defaultFence returns the default fence configuration using OpenAI's
// recommended untrusted_text block format with triple backticks.
func defaultFence() FenceConfig {
	return FenceConfig{
		Start: "```untrusted_text",
		End:   "```",
	}
}

// fence wraps content in fence delimiters to mark it as untrusted.
// This provides a structural defense against prompt injection attacks
// by clearly delineating untrusted content from the rest of the response.
//
// Example output:
//
//	```untrusted_text
//	<actual untrusted content here>
//	```
func fence(content string, config FenceConfig) string {
	return fmt.Sprintf("%s\n%s\n%s", config.Start, content, config.End)
}
