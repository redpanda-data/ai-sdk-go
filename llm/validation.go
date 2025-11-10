package llm

// ConfigValidator defines the interface for provider configuration validation.
// All provider configurations should implement this interface to ensure
// consistent validation behavior across the SDK.
type ConfigValidator interface {
	// Validate checks if the configuration is valid and returns an error if not.
	// This should check all required fields and parameter constraints.
	Validate() error

	// ApplyDefaults sets default values for optional configuration parameters.
	// This should be called before validation to ensure consistent behavior.
	ApplyDefaults()
}
