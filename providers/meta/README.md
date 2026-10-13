# Meta Model API

Direct access to Standard-tier Muse Wonder 1.0 and Muse Spark 1.3, reusing the
SDK's OpenAI Responses transport. No extra dependency or separate protocol
implementation.

Muse Wonder 1.0 is in preview and has different token pricing from Muse Spark
1.3. The root quickstart keeps Muse Spark as its default. See [pricing](https://dev.meta.ai/docs/pricing-rate-limits)
before selecting Wonder.

~~~go
provider, err := meta.NewProvider(os.Getenv("MODEL_API_KEY"))
if err != nil {
    return err
}
model, err := provider.NewModel(
    meta.ModelMuseWonder10, // Preview. Use meta.ModelMuseSpark13 for Muse Spark.
    openai.WithReasoningEffort(openai.ReasoningEffortHigh),
    openai.WithMaxTokens(8192),
)
if err != nil {
    return err
}
// Call model.Generate or model.GenerateEvents with an llm.Request.
~~~

Import github.com/redpanda-data/ai-sdk-go/providers/meta and
github.com/redpanda-data/ai-sdk-go/providers/openai for the shared options.
openai.WithBaseURL, WithHTTPClient, and WithTimeout configure the
provider. The default endpoint is https://api.meta.ai/v1.

## Known limits

- Meta publishes a
  1,048,576-token shared context window, but its docs do not state a separate
  maximum output-token value. The SDK uses a conservative **32,768-token ceiling** for explicitly supplied
  budgets rather than treating the entire shared context as output capacity.
  When no budget is supplied, the field is omitted and the server chooses its default.
  It is not a verified vendor maximum. Catalog attribute
  output_token_limit_source=sdk_conservative_limit records that distinction.
  The server also enforces its own cap and the combined input/output budget.
- Budgets below Meta's documented minimum of 16 are rejected.
- Supported efforts: minimal, low, medium, high, xhigh, max. None is rejected.
- Temperature, output budget, reasoning effort, and reasoning summary use
  the existing Responses mapper. Top-p and sampling penalties are documented
  vendor capabilities but are rejected by this wrapper because the shared
  mapper does not send them; seed is not supported.
- Binary image/audio/video/document input and encrypted reasoning replay are
  not exposed by the shared request mapper. Catalog modalities describe
  vendor capabilities, not transport coverage. Meta warns that 1.3 audio
  understanding quality may be degraded.
- Only Standard tier is registered. Contributor models permit training on
  submitted data and are intentionally excluded. Muse Wonder Full is available
  only in Muse Code, not through Model API. Setting reasoning effort to max
  does not select Muse Wonder Full.
- Native tool search remains disabled in this provider's catalog until the
  shared transport is verified against Meta's hosted tool-search endpoint.
  Meta documents [tool search](https://dev.meta.ai/docs/tool-search), but this
  SDK integration has not been tested with live Meta credentials.
- Live integration and conformance tests require MODEL_API_KEY. Both the
  provider and agent conformance suites run separately for every registered model.

## Verification

~~~sh
go test -short ./providers/meta ./catalog/... ./cmd/catalog-snapshot
MODEL_API_KEY=... go test ./providers/meta -run "Integration|Conformance" -v
~~~

## Sources

- [Models](https://dev.meta.ai/docs/models)
- [Pricing](https://dev.meta.ai/docs/pricing-rate-limits)
- [Reasoning](https://dev.meta.ai/docs/reasoning)
- [Responses API](https://dev.meta.ai/docs/protocols/responses)
- [Muse Spark announcement](https://research.meta.ai/blog/introducing-muse-spark-1-3)
- [Muse Wonder announcement](https://about.fb.com/news/2026/10/introducing-muse-wonder-models/)
