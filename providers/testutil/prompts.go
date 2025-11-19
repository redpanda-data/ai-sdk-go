package testutil

import "strings"

// GenerateLargePrompt generates a large prompt with approximately the target number of tokens.
// Rough estimate: 1 token ≈ 4 characters for English text.
func GenerateLargePrompt(targetTokens int) string {
	const loremIpsum = `Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. `

	// Estimate characters needed (4 chars per token)
	targetChars := targetTokens * 4

	var builder strings.Builder
	builder.WriteString("Context information:\n\n")

	// Repeat lorem ipsum until we hit target
	for builder.Len() < targetChars {
		builder.WriteString(loremIpsum)
	}

	builder.WriteString("\n\nPlease answer the following question based on this context.")

	return builder.String()
}
