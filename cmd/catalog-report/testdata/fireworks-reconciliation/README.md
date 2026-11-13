# Synthetic Fireworks reconciliation cases

These fixtures exercise the maintenance pipeline, not Fireworks inference. The model addition, rate, lifecycle date and capability edits are deliberately synthetic. Their source URLs identify the page classes the report validator permits; the values are not claims about those pages or evidence of a supported model/API contract.

The addition and capability cases follow the ordinary documentation-based catalog rules without live recordings. Recordings are regression fixtures, not catalog qualification certificates; they are not prerequisites for adding documented supported models or features.

The executable test discovers a fixture offering, applies an actual Git patch to the synthetic model source, regenerates the real SDK snapshot, parses and validates the creator's structured report, renders its effects and checks exact replay against stale or dropped patches. No provider credential is read and no inference call is made.
