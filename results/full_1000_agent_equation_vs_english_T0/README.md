# Full 1,000-problem equation-versus-English MCP-agent comparison

## Controlled setup

All five local MCP-agent models from the reused ADF configuration were tested on the same ordered 1,000 corrected SVAMP records. Each model received either the canonical equation or the original English body and question. Temperature (`0`), MCP arithmetic tools, strict numeric scorer, maximum-step policy, and record order were fixed.

## Results

| Model | Equation accuracy | English accuracy | Language gap | Equation latency | English latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| `llama3.2:1b` | 4.5% | 9.4% | -4.9 pp | 2,715 ms | 2,938 ms |
| `llama3.2:3b` | 21.1% | 1.6% | +19.5 pp | 2,197 ms | 2,020 ms |
| `phi4-mini:3.8b` | 66.5% | 0.2% | +66.3 pp | 223 ms | 1,431 ms |
| `qwen3.5:4b` | 99.9% | 78.2% | +21.7 pp | 8,218 ms | 10,389 ms |
| `granite4:3b` | 98.2% | 82.3% | +15.9 pp | 1,413 ms | 1,420 ms |

## Failure and tool-use diagnostics

Each cell combines the model's 1,000 equation and 1,000 English attempts.

| Model | Correct | Wrong | Format failures | API/step failures | Tool errors | No-tool answers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama3.2:1b` | 139/2,000 | 380 | 1375 | 106 | 497 | 42 |
| `llama3.2:3b` | 227/2,000 | 68 | 1702 | 3 | 125 | 104 |
| `phi4-mini:3.8b` | 667/2,000 | 348 | 985 | 0 | 0 | 2000 |
| `qwen3.5:4b` | 1781/2,000 | 6 | 213 | 0 | 0 | 208 |
| `granite4:3b` | 1805/2,000 | 166 | 29 | 0 | 0 | 2 |

## Interpretation

- Granite had the highest English accuracy (82.3%), while Qwen3.5 had the highest equation accuracy (99.9%) and 78.2% English accuracy.
- Both strong models lost accuracy when they had to infer the operation from English instead of receiving the equation directly. This is the language-understanding cost measured by the extension.
- The smaller Llama models produced many format and tool-call failures. Phi4-mini largely answered without calling MCP tools, so its numbers do not demonstrate compliant agent behavior even when the numeric answer was correct.

This is a controlled model-and-input comparison, not a claim about all language understanding. It uses English only; the separate 20-record multilingual pilot remains preliminary because its translations have not been human-reviewed.

The ten raw JSON files in this directory contain all 10,000 per-record outcomes and tool traces.
