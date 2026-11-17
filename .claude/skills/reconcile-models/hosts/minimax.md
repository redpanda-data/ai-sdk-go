# MiniMax models

MiniMax is the publisher (`minimax`), not Fireworks. This run currently owns MiniMax's copies in `providers/fireworks`; there is no direct MiniMax provider package. Start with `shared.md` and `fireworks.md`.

## Pages

| Fact | Page | Where on it |
|---|---|---|
| Model-wide description and thinking modes | `https://huggingface.co/MiniMaxAI/MiniMax-M3/raw/main/README.md` | Introduction and How to Use: native multimodality and enabled/adaptive/disabled modes |
| Native architecture limits | `https://huggingface.co/MiniMaxAI/MiniMax-M3/raw/main/config.json` | `text_config.max_position_embeddings`; this is not Fireworks' smaller serving window |
| Hosted model, serving limits and release evidence | `https://fireworks.ai/models/fireworks/minimax-m3` | Metadata, Supporting Functionality and FAQ |
| Fireworks reasoning mapping | `https://docs.fireworks.ai/api-reference/post-chatcompletions.md` | `reasoning_effort` and `ThinkingConfigAdaptive`, which explicitly names M3's adaptive default |

## Reading them

- Creator checkpoint pages are only those authored under `huggingface.co/MiniMaxAI/`; third-party cards and quantizations are not creator authority. Repository creation is not the first public model release.
- M3's creator supports enabled/adaptive/disabled thinking, and Fireworks explicitly documents adaptive for M3. Do not inherit MiniMax M2's always-on low/medium/high map, advertise graded effort semantics from a shared enum, or invent a model-specific hard thinking budget.
- Native M3 is multimodal with 1,048,576 context. Fireworks' `minimax-m3` page currently describes a smaller 512K served window and no image input. Host capabilities are not automatically the creator's full modality set.
- The Fireworks FAQ gives a model-wide release of 2026-06-01, while its resource was created 2026-06-11. The creator README supplies no separate reliable knowledge cutoff; absence stays unknown.
- Current cache rates come from Fireworks' central pricing table, not the model FAQ's stale $0.059 number. Prices, availability, lifecycle and wire bindings are Fireworks host facts. Its US-only retirement does not retire the global offering.
