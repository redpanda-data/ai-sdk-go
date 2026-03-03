package agentpack

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go/bedrock"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/google"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/openaicompat"
)

const (
	defaultOllamaBaseURL = "http://localhost:11434"
)

// newModel creates an LLM model from environment variables.
//
// Required env vars:
//   - AI_PROVIDER: anthropic | openai | google | openai-compat | ollama | bedrock
//   - AI_MODEL: model name
//
// Optional env vars:
//   - AI_API_KEY: API key (falls back to provider-specific vars; not required for ollama/bedrock)
//   - AI_BASE_URL: custom endpoint
//   - AWS_REGION: AWS region for bedrock (uses AWS SDK default config)
func newModel(ctx context.Context) (llm.Model, error) {
	providerName := os.Getenv("AI_PROVIDER")
	if providerName == "" {
		return nil, fmt.Errorf("AI_PROVIDER env var is required")
	}

	modelName := os.Getenv("AI_MODEL")
	if modelName == "" {
		return nil, fmt.Errorf("AI_MODEL env var is required")
	}

	apiKey := resolveAPIKey(providerName)
	baseURL := os.Getenv("AI_BASE_URL")

	switch providerName {
	case "anthropic":
		return newAnthropicModel(apiKey, baseURL, modelName)
	case "bedrock":
		return newBedrockModel(ctx, modelName)
	case "openai":
		return newOpenAIModel(apiKey, baseURL, modelName)
	case "google":
		return newGoogleModel(ctx, apiKey, baseURL, modelName)
	case "openai-compat":
		return newOpenAICompatModel(apiKey, baseURL, modelName, "openaicompat")
	case "ollama":
		return newOllamaModel(baseURL, modelName)
	default:
		return nil, fmt.Errorf("unknown AI_PROVIDER: %s", providerName)
	}
}

func resolveAPIKey(providerName string) string {
	if key := os.Getenv("AI_API_KEY"); key != "" {
		return key
	}
	switch providerName {
	case "anthropic":
		return os.Getenv("ANTHROPIC_API_KEY")
	case "openai":
		return os.Getenv("OPENAI_API_KEY")
	case "google":
		return os.Getenv("GOOGLE_API_KEY")
	}
	return ""
}

func newAnthropicModel(apiKey, baseURL, modelName string) (llm.Model, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("AI_API_KEY (or ANTHROPIC_API_KEY) is required for anthropic provider")
	}
	var opts []anthropic.ProviderOption
	if baseURL != "" {
		opts = append(opts, anthropic.WithBaseURL(baseURL))
	}
	p, err := anthropic.NewProvider(apiKey, opts...)
	if err != nil {
		return nil, fmt.Errorf("create anthropic provider: %w", err)
	}
	m, err := p.NewModel(modelName)
	if err != nil {
		return nil, fmt.Errorf("create anthropic model %s: %w", modelName, err)
	}
	return m, nil
}

func newOpenAIModel(apiKey, baseURL, modelName string) (llm.Model, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("AI_API_KEY (or OPENAI_API_KEY) is required for openai provider")
	}
	var opts []openai.ProviderOption
	if baseURL != "" {
		opts = append(opts, openai.WithBaseURL(baseURL))
	}
	p, err := openai.NewProvider(apiKey, opts...)
	if err != nil {
		return nil, fmt.Errorf("create openai provider: %w", err)
	}
	m, err := p.NewModel(modelName)
	if err != nil {
		return nil, fmt.Errorf("create openai model %s: %w", modelName, err)
	}
	return m, nil
}

func newGoogleModel(ctx context.Context, apiKey, baseURL, modelName string) (llm.Model, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("AI_API_KEY (or GOOGLE_API_KEY) is required for google provider")
	}
	var opts []google.ProviderOption
	if baseURL != "" {
		opts = append(opts, google.WithBaseURL(baseURL))
	}
	p, err := google.NewProvider(ctx, apiKey, opts...)
	if err != nil {
		return nil, fmt.Errorf("create google provider: %w", err)
	}
	m, err := p.NewModel(modelName)
	if err != nil {
		return nil, fmt.Errorf("create google model %s: %w", modelName, err)
	}
	return m, nil
}

func newOpenAICompatModel(apiKey, baseURL, modelName, providerName string) (llm.Model, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("AI_API_KEY is required for openai-compat provider")
	}
	var opts []openaicompat.ProviderOption
	if baseURL != "" {
		opts = append(opts, openaicompat.WithBaseURL(baseURL))
	}
	if providerName != "" {
		opts = append(opts, openaicompat.WithProviderName(providerName))
	}
	p, err := openaicompat.NewProvider(apiKey, opts...)
	if err != nil {
		return nil, fmt.Errorf("create openai-compat provider: %w", err)
	}
	m, err := p.NewModel(modelName)
	if err != nil {
		return nil, fmt.Errorf("create openai-compat model %s: %w", modelName, err)
	}
	return m, nil
}

func newBedrockModel(ctx context.Context, modelName string) (llm.Model, error) {
	// bedrock.WithLoadDefaultConfig uses AWS SDK default credential chain
	// (env vars AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, IAM roles, SSO, etc.)
	// and reads AWS_REGION for the endpoint.
	bedrockOpt := bedrock.WithLoadDefaultConfig(ctx)

	var providerOpts []anthropic.ProviderOption
	providerOpts = append(providerOpts, anthropic.WithRequestOptions(bedrockOpt))

	// When LOG_LEVEL=debug, wrap the HTTP client to log request/response details.
	if strings.ToLower(os.Getenv("LOG_LEVEL")) == "debug" {
		providerOpts = append(providerOpts, anthropic.WithHTTPClient(&http.Client{
			Timeout:   10 * time.Minute,
			Transport: &debugTransport{base: http.DefaultTransport},
		}))
	}

	p, err := anthropic.NewProvider("bedrock", providerOpts...) // dummy key — Bedrock uses AWS credentials
	if err != nil {
		return nil, fmt.Errorf("create bedrock provider: %w", err)
	}
	m, err := p.NewModel(modelName)
	if err != nil {
		return nil, fmt.Errorf("create bedrock model %s: %w", modelName, err)
	}
	return m, nil
}

// debugTransport logs HTTP request/response details for debugging API issues.
type debugTransport struct {
	base http.RoundTripper
}

func (d *debugTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	fmt.Fprintf(os.Stderr, "[DEBUG] %s %s\n", req.Method, req.URL)
	for k, v := range req.Header {
		if k == "Authorization" || k == "X-Api-Key" {
			fmt.Fprintf(os.Stderr, "[DEBUG]   %s: [redacted]\n", k)
		} else {
			fmt.Fprintf(os.Stderr, "[DEBUG]   %s: %s\n", k, v)
		}
	}
	resp, err := d.base.RoundTrip(req)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[DEBUG] Transport error: %v\n", err)
		return resp, err
	}
	fmt.Fprintf(os.Stderr, "[DEBUG] Response: %d %s\n", resp.StatusCode, resp.Status)
	fmt.Fprintf(os.Stderr, "[DEBUG]   Content-Type: %s\n", resp.Header.Get("Content-Type"))
	return resp, err
}

func newOllamaModel(baseURL, modelName string) (llm.Model, error) {
	if baseURL == "" {
		baseURL = defaultOllamaBaseURL
	}
	p, err := openaicompat.NewProvider("ollama",
		openaicompat.WithBaseURL(baseURL),
		openaicompat.WithProviderName("ollama"),
	)
	if err != nil {
		return nil, fmt.Errorf("create ollama provider: %w", err)
	}
	m, err := p.NewModel(modelName)
	if err != nil {
		return nil, fmt.Errorf("create ollama model %s: %w", modelName, err)
	}
	return m, nil
}
