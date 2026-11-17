# Z.ai models

Z.ai is the publisher (`zai`), not Fireworks. This run currently owns Z.ai's copies in `providers/fireworks`; there is no direct Z.ai provider package. Start with `shared.md` and `fireworks.md`.

## Pages

| Fact | Page | Where on it |
|---|---|---|
| GLM 5.3 model and reasoning | `https://docs.z.ai/guides/llm/glm-5.3.md` | Feature Changes: native context/output limits, always-on thinking and low/high/max efforts |
| GLM 5.3 Flash model and reasoning | `https://docs.z.ai/guides/llm/glm-5.3-flash.md` | Model Overview, maximum output and Parameter Settings; native multimodal model |
| Release days | `https://z.ai/blog/glm-5.3` and `https://z.ai/blog/glm-5.3-flash` | announcement date and availability wording, not the later weight release or repository creation |
| Creator checkpoints | `https://huggingface.co/zai-org/GLM-5.3/raw/main/README.md` and `https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/README.md` | Z.ai-authored model descriptions and reasoning notes; each repository's `config.json` gives native architecture limits |

## Reading them

- The blog is client-rendered. Read its HTML to find the linked `/blog/assets/glm-5.3-*.js` or `/blog/assets/glm-5.3-flash-*.js` module, then fetch that exact module in a separate plain `curl` command. Its rendered date and availability prose are the source, not the module filename. GLM 5.3's announcement is 2026-08-14; Flash's is 2026-08-26.
- Creator checkpoint pages are only those authored under `huggingface.co/zai-org/`; third-party cards and quantizations are not creator authority.
- GLM 5.3 and Flash publish low/high/max effort, default max, with thinking always enabled. Do not copy the older GLM binary on/off map or infer medium support from a shared API enum.
- The creator API publishes 1M context and 128K output; Fireworks' serving limits and API-specific controls still come from Fireworks, subject to RULES.md (capped-at-vendor, limit-follow, reasoning-unsendable).
- Canonical model IDs name Z.ai. Fireworks invocation IDs use `glm-5p3` and `glm-5p3-flash`; dots in creator IDs and `p` in host IDs are not interchangeable aliases.
- Fireworks' prices, lifecycle and supported wire bindings are host facts. Do not use Z.ai's own API prices or endpoint compatibility to change those copies.
