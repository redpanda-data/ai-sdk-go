package webfetch

import (
	"time"
)

// Config holds the configuration for the webfetch tool.
type Config struct {
	Timeout           time.Duration // Request timeout
	MaxRedirects      int           // Maximum number of redirects to follow
	MaxResponseBytes  int64         // Maximum response body size
	AllowedSchemes    []string      // Allowed URL schemes (default: ["https"])
	AllowedPorts      []int         // Allowed ports (default: [443])
	DenyPrivateIPs    bool          // Block private/reserved IP addresses (default: true)
	AllowedMedia      []string      // Allowed media types (supports wildcards like "text/*")
	ConvertToMarkdown bool          // Convert HTML content to markdown (default: true)
	Fencing           bool          // Enable prompt injection protection fencing (default: true)
	FenceConfig       FenceConfig   // Fence delimiters for untrusted content (default: OpenAI's untrusted_text format)
}

// DefaultConfig returns a configuration with secure defaults.
func DefaultConfig() Config {
	return Config{
		Timeout:           10 * time.Second,
		MaxRedirects:      5,
		MaxResponseBytes:  2 * 1024 * 1024, // 2MB
		AllowedSchemes:    []string{"https"},
		AllowedPorts:      []int{443},
		DenyPrivateIPs:    true,
		ConvertToMarkdown: true,           // Enable markdown conversion by default
		Fencing:           true,           // Fencing enabled by default (secure-by-default)
		FenceConfig:       defaultFence(), // Use OpenAI's recommended untrusted_text format
		AllowedMedia: []string{
			"text/*",
			"application/json",
			"application/xml",
			"application/xhtml+xml",
		},
	}
}

// Option is a functional option for configuring the webfetch tool.
type Option func(*Config)

// WithTimeout sets the request timeout.
func WithTimeout(timeout time.Duration) Option {
	return func(cfg *Config) {
		cfg.Timeout = timeout
	}
}

// WithMaxBytes sets the maximum response body size.
func WithMaxBytes(maxBytes int64) Option {
	return func(cfg *Config) {
		cfg.MaxResponseBytes = maxBytes
	}
}

// WithMaxRedirects sets the maximum number of redirects to follow.
func WithMaxRedirects(maxRedirects int) Option {
	return func(cfg *Config) {
		cfg.MaxRedirects = maxRedirects
	}
}

// WithAllowedSchemes sets the allowed URL schemes.
func WithAllowedSchemes(schemes []string) Option {
	return func(cfg *Config) {
		cfg.AllowedSchemes = make([]string, len(schemes))
		copy(cfg.AllowedSchemes, schemes)
	}
}

// WithAllowedPorts sets the allowed ports.
func WithAllowedPorts(ports []int) Option {
	return func(cfg *Config) {
		cfg.AllowedPorts = make([]int, len(ports))
		copy(cfg.AllowedPorts, ports)
	}
}

// WithDenyPrivateIPs controls whether private IP addresses are blocked.
func WithDenyPrivateIPs(deny bool) Option {
	return func(cfg *Config) {
		cfg.DenyPrivateIPs = deny
	}
}

// WithAllowedMedia sets the allowed media types.
func WithAllowedMedia(mediaTypes []string) Option {
	return func(cfg *Config) {
		cfg.AllowedMedia = make([]string, len(mediaTypes))
		copy(cfg.AllowedMedia, mediaTypes)
	}
}

// WithConvertToMarkdown controls whether HTML content is converted to markdown.
func WithConvertToMarkdown(convert bool) Option {
	return func(cfg *Config) {
		cfg.ConvertToMarkdown = convert
	}
}

// WithFencing enables or disables prompt injection protection fencing.
// By default, fencing is enabled to protect against prompt injection attacks.
func WithFencing(enabled bool) Option {
	return func(cfg *Config) {
		cfg.Fencing = enabled
	}
}

// WithCustomFence sets custom fence delimiters for untrusted content.
// The start and end parameters define the opening and closing delimiters.
// By default, OpenAI's recommended format is used: "```untrusted_text" / "```".
func WithCustomFence(start, end string) Option {
	return func(cfg *Config) {
		cfg.FenceConfig = FenceConfig{
			Start: start,
			End:   end,
		}
	}
}
