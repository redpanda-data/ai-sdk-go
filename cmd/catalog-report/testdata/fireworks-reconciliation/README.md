# Synthetic Fireworks reconciliation cases

These fixtures exercise the maintenance pipeline, not Fireworks inference. The rates, limits, lifecycle date and alias edits are deliberately synthetic. Their source URLs identify the page classes the report validator permits; the values are not claims about those pages or evidence of a supported model/API contract.

The addition and alias cases assume an already-qualified contract in the test scenario. They do not qualify new live offerings or aliases. The real catalog's committed live-contract manifest remains responsible for exact offering/API/feature completeness, and unqualified expansions follow the Fireworks qualification Ask.

The executable test discovers a fixture offering, applies an actual Git patch to the synthetic model source, regenerates the real SDK snapshot, parses and validates the creator's structured report, renders its effects and checks exact replay against stale or dropped patches. No provider credential is read and no inference call is made.
