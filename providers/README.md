# Providers

This directory contains provider implementations for various LLM services.

## Available Providers

- **anthropic** - Anthropic Claude models (Sonnet, Haiku, Opus)
- **openai** - OpenAI models (GPT-5.2, GPT-5.1, GPT-5, GPT-4.x, o-series reasoning models)
- **gemini** - Google Gemini models (2.5 Pro, 2.5 Flash, etc.)
- **openaicompat** - Generic OpenAI-compatible API provider

## Model Constraints Reference

Model constraints (context window sizes, output token limits) are based on data from LiteLLM's comprehensive model database:

**Reference:** https://raw.githubusercontent.com/BerriAI/litellm/refs/heads/main/model_prices_and_context_window.json

This JSON file contains authoritative information about:
- `max_input_tokens` - Maximum context window size
- `max_output_tokens` - Maximum output token limit
- Token pricing and other model metadata

When adding new models or updating existing constraints, refer to this file for accurate specifications.
