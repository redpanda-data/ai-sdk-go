package agentpack

import (
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

// AgentConfig represents the parsed agent.yaml manifest.
type AgentConfig struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description"`

	// Prompt: inline or from file (one of these must be set).
	Prompt     string `yaml:"prompt"`
	PromptFile string `yaml:"prompt_file"`

	// Agent tuning (optional).
	MaxTurns        int `yaml:"max_turns"`
	ToolConcurrency int `yaml:"tool_concurrency"`

	// MCP servers (optional).
	MCPs map[string]MCPConfig `yaml:"mcps"`
}

// MCPConfig defines an MCP server connection.
type MCPConfig struct {
	// Stdio transport.
	Command string   `yaml:"command"`
	Args    []string `yaml:"args"`
	Env     map[string]string `yaml:"env"`

	// HTTP/SSE transport.
	URL     string            `yaml:"url"`
	Headers map[string]string `yaml:"headers"`
}

// IsStdio returns true if this MCP config uses stdio transport.
func (m MCPConfig) IsStdio() bool {
	return m.Command != ""
}

// LoadConfig reads and parses an agent.yaml file.
func LoadConfig(path string) (*AgentConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config %s: %w", path, err)
	}

	return ParseConfig(data)
}

// ParseConfig parses YAML bytes into an AgentConfig.
func ParseConfig(data []byte) (*AgentConfig, error) {
	// Expand ${VAR} references in the YAML before parsing.
	expanded := os.ExpandEnv(string(data))

	var cfg AgentConfig
	if err := yaml.Unmarshal([]byte(expanded), &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	if err := cfg.validate(); err != nil {
		return nil, err
	}

	return &cfg, nil
}

// ResolvePrompt returns the final prompt string, loading from file if needed.
// The promptFile path is resolved relative to the given base directory.
func (c *AgentConfig) ResolvePrompt(baseDir string) (string, error) {
	if c.Prompt != "" {
		return c.Prompt, nil
	}

	path := c.PromptFile
	if !strings.HasPrefix(path, "/") {
		path = baseDir + "/" + path
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read prompt file %s: %w", path, err)
	}

	return strings.TrimSpace(string(data)), nil
}

func (c *AgentConfig) validate() error {
	if c.Name == "" {
		return fmt.Errorf("config: name is required")
	}
	if c.Prompt == "" && c.PromptFile == "" {
		return fmt.Errorf("config: prompt or prompt_file is required")
	}
	if c.Prompt != "" && c.PromptFile != "" {
		return fmt.Errorf("config: only one of prompt or prompt_file may be set")
	}
	return nil
}
