package agentpack

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseConfig_InlinePrompt(t *testing.T) {
	yaml := `
name: test-agent
description: A test agent
prompt: You are a helpful assistant.
`
	cfg, err := ParseConfig([]byte(yaml))
	require.NoError(t, err)
	assert.Equal(t, "test-agent", cfg.Name)
	assert.Equal(t, "A test agent", cfg.Description)
	assert.Equal(t, "You are a helpful assistant.", cfg.Prompt)
}

func TestParseConfig_PromptFile(t *testing.T) {
	yaml := `
name: test-agent
prompt_file: prompt.md
`
	cfg, err := ParseConfig([]byte(yaml))
	require.NoError(t, err)
	assert.Equal(t, "prompt.md", cfg.PromptFile)
}

func TestParseConfig_WithMCPs(t *testing.T) {
	yaml := `
name: test-agent
prompt: hello
mcps:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_TOKEN: my-token
  slack:
    url: https://slack-mcp.example.com/sse
    headers:
      Authorization: Bearer my-slack-token
`
	cfg, err := ParseConfig([]byte(yaml))
	require.NoError(t, err)

	require.Len(t, cfg.MCPs, 2)

	gh := cfg.MCPs["github"]
	assert.True(t, gh.IsStdio())
	assert.Equal(t, "npx", gh.Command)
	assert.Equal(t, []string{"-y", "@modelcontextprotocol/server-github"}, gh.Args)
	assert.Equal(t, "my-token", gh.Env["GITHUB_TOKEN"])

	sl := cfg.MCPs["slack"]
	assert.False(t, sl.IsStdio())
	assert.Equal(t, "https://slack-mcp.example.com/sse", sl.URL)
	assert.Equal(t, "Bearer my-slack-token", sl.Headers["Authorization"])
}

func TestParseConfig_EnvVarExpansion(t *testing.T) {
	t.Setenv("TEST_TOKEN", "secret-123")

	yaml := `
name: test-agent
prompt: hello
mcps:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_TOKEN: ${TEST_TOKEN}
`
	cfg, err := ParseConfig([]byte(yaml))
	require.NoError(t, err)
	assert.Equal(t, "secret-123", cfg.MCPs["github"].Env["GITHUB_TOKEN"])
}

func TestParseConfig_OptionalTuning(t *testing.T) {
	yaml := `
name: test-agent
prompt: hello
max_turns: 10
tool_concurrency: 5
`
	cfg, err := ParseConfig([]byte(yaml))
	require.NoError(t, err)
	assert.Equal(t, 10, cfg.MaxTurns)
	assert.Equal(t, 5, cfg.ToolConcurrency)
}

func TestParseConfig_MissingName(t *testing.T) {
	yaml := `
prompt: hello
`
	_, err := ParseConfig([]byte(yaml))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "name is required")
}

func TestParseConfig_MissingPrompt(t *testing.T) {
	yaml := `
name: test-agent
`
	_, err := ParseConfig([]byte(yaml))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "prompt or prompt_file is required")
}

func TestParseConfig_BothPromptAndPromptFile(t *testing.T) {
	yaml := `
name: test-agent
prompt: hello
prompt_file: prompt.md
`
	_, err := ParseConfig([]byte(yaml))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "only one of prompt or prompt_file")
}

func TestResolvePrompt_Inline(t *testing.T) {
	cfg := &AgentConfig{Prompt: "You are helpful."}
	prompt, err := cfg.ResolvePrompt("/tmp")
	require.NoError(t, err)
	assert.Equal(t, "You are helpful.", prompt)
}

func TestResolvePrompt_FromFile(t *testing.T) {
	dir := t.TempDir()
	promptPath := filepath.Join(dir, "prompt.md")
	err := os.WriteFile(promptPath, []byte("You are a test assistant.\n"), 0644)
	require.NoError(t, err)

	cfg := &AgentConfig{PromptFile: "prompt.md"}
	prompt, err := cfg.ResolvePrompt(dir)
	require.NoError(t, err)
	assert.Equal(t, "You are a test assistant.", prompt)
}

func TestResolvePrompt_FileNotFound(t *testing.T) {
	cfg := &AgentConfig{PromptFile: "nonexistent.md"}
	_, err := cfg.ResolvePrompt("/tmp")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "read prompt file")
}

func TestLoadConfig(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "agent.yaml")
	err := os.WriteFile(configPath, []byte(`
name: file-agent
prompt: Hello from file
`), 0644)
	require.NoError(t, err)

	cfg, err := LoadConfig(configPath)
	require.NoError(t, err)
	assert.Equal(t, "file-agent", cfg.Name)
	assert.Equal(t, "Hello from file", cfg.Prompt)
}
