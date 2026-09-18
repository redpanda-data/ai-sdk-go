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
| — | `provider.Catalog().Replacement(offeringID)` (announced `ReplacedBy`, else the series successor) and `.Offerings(modelID)` |
| — | `provider.Catalog().ResolveID(name)` — `Resolve` without the offering copy, for hot paths |

`llm.ModelDiscoveryInfo` and every `Models()` method are removed.
`openaicompat` has no static catalog; its `Catalog()` returns nil
(model names there are caller-defined).

### Pricing

| Old | New |
|---|---|
| `provider.ModelPricing() map[string]pricing.Info` | `provider.Catalog().PricingByID()` |
| `Lookup(modelID) (Info, bool)` | `Lookup(provider, modelID) (Info, error)` |
| `Calculate(modelID, …)` | `Calculate(provider, modelID, …)` |
| `WithOverride(modelID, info)` | `WithOverride(provider, modelID, info)` |
| `WithProvider(prov.Name(), prov.Catalog().PricingByID())` | `WithSource(prov.Catalog())` |
| `WithProvider(provider string, …)` | `WithProvider(provider pricing.ProviderKey, …)` — kept for callers holding a bare map |
| — | `pricing.ProviderKey` — the provider half of every catalog key |
| — | `pricing.ErrUnknownProvider` — separable from `ErrUnknownModel`; alert on it, don't just log |
| — | `openai.ProviderName` (also `anthropic.`, `google.`, `bedrock.`, `vertex.ProviderName`) — the key as a const |

`PricingByID` includes official aliases (e.g. OpenAI's `"gpt-5.6"`), so
exact-ID billing lookups keep working. Timestamped snapshot IDs (as
reported in `llm.Response.InvokedModelID`) are not enumerable — resolve
them first:

```go
off, ok := openai.Catalog().Resolve(resp.InvokedModelID)
if !ok {
    // unknown model: treat as UNPRICED, never as free
}
// Calculate and Lookup now take the ProviderKey first: the same bare
// model ID can carry a different rate card per provider.
//
// Take the key from the provider's ProviderName const (untyped string,
// no cast); a hand-typed literal is the one break the compiler cannot catch.
cost, err := priceCat.Calculate(openai.ProviderName, off.ID, resp.Usage, req)
switch {
case errors.Is(err, pricing.ErrUnknownProvider):
    // Mapping bug — every call at this site prices at $0. Alert, don't just log.
case errors.Is(err, pricing.ErrUnknownModel):
    // New model under a known provider: treat as UNPRICED, never as free.
case err != nil:
    // other error
}
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
- **Only version stamps prefix-match.** A suffix resolves to its family
  only when it is a date or revision stamp (`-20250929`, `-2025-04-16`,
  `-001`, `@001`). Short version bumps such as `"gpt-5.7"` or
  `"claude-opus-5-1"` are unknown models and `NewModel` rejects them,
  where the old resolver silently mapped them onto `gpt-5` /
  `claude-opus-5` with that model's constraints and pricing.
- **Retired models stay in the catalog.** The catalog is append-only:
  retired offerings remain (with `Life.Retires` in the past and a
  `Deprecated:` marker on their ID constants) so historical usage stays
  priceable. Use `Catalog().Now()` views to filter by lifecycle.
- **`llm.Response.InvokedModelID`** now reports catalog offering IDs
  where the provider reports a snapshot the catalog recognises;
  unrecognised IDs pass through unchanged.
- **The Google catalog's provider key is now `gcp.gemini`, not
  `google`** — in Go, and in `catalog/snapshot.json`, whose `provider`
  field read `google`. It is the value `google.Catalog().Provider()` and
  `google.ProviderName` return. A hand-typed `google` misses every Gemini
  lookup and prices it at $0. Non-Go consumers that persisted or filter on
  `"google"` must rewrite it before bumping. Vertex's key is `gcp.vertex`.
- **Vertex offering IDs are the bare publisher IDs** (`claude-sonnet-5`,
  not `vertex.claude-sonnet-5`), and `vertex.Offering*` is now
  `vertex.Model*`. `OfferingForModel`, `LocationsForModel` and
  `IsModelAvailableAtLocation` no longer strip a `vertex.` prefix: they
  take a `string`, so a caller still passing the prefixed form compiles
  clean and gets `ok == false` / `nil` / `false` — a silently missing
  model, not a compile error. `catalog/snapshot.json` is the read format
  for non-Go consumers, so these IDs change under them. Consumers that
  persisted a prefixed ID must rewrite it before bumping; after the
  first stored row this becomes a data migration, not a revert.
- **`vertex.ModelMetadataVertexModel` and its `vertex_model` attribute are
  gone.** The attribute held the bare wire model ID, which was the offering
  ID with the `vertex.` prefix stripped. Now that the prefix is gone the
  offering ID is already that bare ID, so the attribute duplicated it on
  every entry. Read `Offering.ID` instead. Go consumers get a compile error
  on the removed constant; non-Go consumers lose the `vertex_model` entry
  from each Vertex offering's `attributes` list in `catalog/snapshot.json`,
  which the tolerant-reader contract already required them not to depend on.
- **`catalog/snapshot.json` is now `schema_version` 2.** The field shape
  did not change; the value domain of `id` did. A model ID is no longer
  unique across the snapshot — `claude-sonnet-5` appears under both
  `anthropic` and `gcp.vertex` with different rate cards — so a consumer
  MUST key an offering by `{provider, id}` and never by `id` alone. The
  tolerant-reader contract does not cover this: ignoring unknown fields
  does not help a consumer that holds one entry per ID. The shared
  `facts` map stays keyed by model ID, because facts are
  provider-independent and `Encode` rejects a conflict across providers.
  This is the exported `snapshot.SchemaVersion`, which is a different
  constant from the pricing hash seed below.
- **`Cost.CatalogVersion` changes at this release.** The hash seed moved
  from `v1` to `v2` — the unexported `pricing.schemaVersion`, not the
  snapshot version above — and that alone makes every `Version()` differ
  even where the rate data is byte-identical; the hash now also folds in
  the provider and the known-provider set. Stored versions from before
  the bump stay valid for historical rows and will not recur.

### Unknown models

`Catalog().Resolve` returning `ok == false` means the catalog does not
know the model. Treat unknown as *stop enforcing* — do not assume
capabilities, constraints, or pricing for it, and never bill it as
free.

On the pricing side there are now two distinct misses, and an
`ErrUnknownModel` guard alone does not cover both. A wrong `ProviderKey`
— a hand-typed `"google"` where the catalog registered `"gcp.gemini"`,
or any provider the catalog carries no rates for — surfaces as
`ErrUnknownProvider`, not `ErrUnknownModel`. Code that only checks for
`ErrUnknownModel` treats that mapping bug as some other error and can let
every call at the site price at $0. Branch on `ErrUnknownProvider` first
and alert on it; see the `Calculate` example above.
