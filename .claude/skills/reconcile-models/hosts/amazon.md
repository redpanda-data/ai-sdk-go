# Amazon (vendor of Nova)

Amazon's own pages for its models. Amazon has no package of its own: Bedrock sells its models, so this run owns them in `providers/bedrock` (RULES.md, Report values, says how to name them). Amazon publishes its models' facts only on their Bedrock cards, which `bedrock.md` lists under "Amazon's model cards". Start with `shared.md`.

## Reading the cards

- The list holds Nova chat models and Titan, embedding, image, video and speech models.
- Bedrock is Amazon's API for its models, so `Released` is the card's Model launch date (released-first-callable) and `Knowledge` its Knowledge cutoff (knowledge-month-end).
