# Fixed multilingual SVAMP corpus

This folder holds the fixed, auditable translations used by the multilingual
extension of the ADF experiment. It intentionally contains no answer labels
or equations: evaluation always uses the unchanged records in
`upstream_adf/data/`.

## Languages

The experiment uses the six official UN languages:

| Code | Language | File |
| --- | --- | --- |
| `en` | English | No translation file; uses the unchanged SVAMP `Body` and `Question`. |
| `ar` | Arabic | `svamp_ar.json` |
| `zh` | Chinese (Simplified) | `svamp_zh.json` |
| `fr` | French | `svamp_fr.json` |
| `ru` | Russian | `svamp_ru.json` |
| `es` | Spanish | `svamp_es.json` |

Each non-English file must have this structure:

```json
{
  "schema_version": 1,
  "metadata": {
    "language": "es",
    "source_language": "en",
    "translation_method": "record the fixed translation model/service and settings here",
    "source_rows_sha256": "hash of the ordered 1,000-row English source",
    "review_status": "reviewed"
  },
  "records": [
    {"ID": "chal-777", "Body": "...", "Question": "..."}
  ]
}
```

The records must contain all 1,000 IDs in the same order as the concatenated
upstream `train.json` and `test.json` files. They must preserve every numeric
literal in the original body and question. The loader enforces these
requirements before a run begins.

## Why translations are fixed before evaluation

Translation at inference time would introduce an unmeasured translation model
into each model call and make a run hard to reproduce. Instead, generate,
inspect, and freeze the five non-English files before evaluating any target
model. `extension/prepare_multilingual_svamp.py` provides a reproducible
optional NLLB generation workflow; its output still needs a documented quality
check before it is treated as benchmark data. After that check, set
`metadata.review_status` to `reviewed`; the experiment loader rejects any
other value by default. A small exploratory run may explicitly pass
`--allow-machine-generated-translations`; the saved result is then marked
`preliminary_machine_translated` and must not be presented as a validated
cross-language benchmark.
