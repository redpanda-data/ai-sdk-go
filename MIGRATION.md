# Migration guide

## Model catalog rework (next release)

The per-provider model metadata surfaces are replaced by one shared,
validated catalog per provider: `catalog.Catalog`. It carries everything
the old surfaces did, plus modalities, reasoning controls, lifecycle
(deprecation/retirement dates, announced replacements), and pricing —
with derived classification (current vs previous
generation, retired, price tier) computed at read time.

All changes below land in a single breaking release; no release is
tagged mid-sequence.

### Discovery

| Old | New |
|---|---|
| `provider.Models() []llm.ModelDiscoveryInfo` | `provider.Catalog().All() []catalog.Offering` |
| `llm.ModelDiscoveryInfo.Name` | `catalog.Offering.ID` |
| `llm.ModelDiscoveryInfo.Metadata["..."]` | `catalog.Offering.Attributes["..."]` |
| — | `provider.Catalog().Now().Current()` / `.Previous()` / `.Deprecated()` / `.Retired()` |

`llm.ModelDiscoveryInfo` and every `Models()` method are removed.
`openaicompat` has no static catalog; its `Catalog()` returns nil
(model names there are caller-defined).

### Pricing

| Old | New |
|---|---|
| `provider.ModelPricing() map[string]pricing.Info` | `provider.Catalog().PricingByID()` |

`PricingByID` includes official aliases (e.g. OpenAI's `"gpt-5.6"`), so
exact-ID billing lookups keep working. Timestamped snapshot IDs (as
reported in `llm.Response.InvokedModelID`) are not enumerable — resolve
them first:

```go
off, ok := openai.Catalog().Resolve(resp.InvokedModelID)
if !ok {
    // unknown model: treat as UNPRICED, never as free
}
cost, err := priceCat.Calculate(off.ID, resp.Usage, req)
```

### Authoring / provider-specific types

| Old | New |
|---|---|
| `openai.ModelDefinition` (also `anthropic.`, `google.`, `bedrock.ModelDefinition`) | authored `catalog.Entry` values inside each provider package |
| `openai.NewCompatModel(name, openai.ModelDefinition, opts...)` | `openai.NewCompatModel(name, openai.CompatModelDefinition{Capabilities, Constraints, Reasoning}, opts...)` |
| `bedrock.ThinkingSupport` | `catalog.ReasoningSupport` (same shape: `Efforts`, `Adaptive`, `Budget`) |
| `anthropic.ModelDefinition.AdaptiveThinking` | `catalog.Offering.Reasoning.Adaptive` |
| `anthropic.ModelDefinition.SupportedSpeeds` | `catalog.Offering.Speeds` |

`CompatModelDefinition` is deliberately the transport subset — the three
fields the OpenAI request path actually consumes — not the catalog
shape.

### Behavior changes

- **Stricter reasoning-effort validation (OpenAI).** A requested effort
  against a model with no declared efforts is now rejected; previously
  it was silently accepted. This matches the other providers.
- **Alias-aware resolution.** `NewModel` resolves official aliases and
  their suffixed forms (`"gpt-5.6-<snapshot>"` → `gpt-5.6-sol`), and
  Google's `NewModel` now accepts versioned variants
  (`"gemini-2.5-flash-001"`, `"models/..."`) that previously required
  the exact family ID.
- **Retired models stay in the catalog.** The catalog is append-only:
  retired offerings remain (with `Life.Retires` in the past and a
  `Deprecated:` marker on their ID constants) so historical usage stays
  priceable. Use `Catalog().Now()` views to filter by lifecycle.
- **`llm.Response.InvokedModelID`** now reports catalog offering IDs
  where the provider reports a snapshot the catalog recognises;
  unrecognised IDs pass through unchanged.

### Unknown models

`Catalog().Resolve` returning `ok == false` means the catalog does not
know the model. Treat unknown as *stop enforcing* — do not assume
capabilities, constraints, or pricing for it, and never bill it as
free.
