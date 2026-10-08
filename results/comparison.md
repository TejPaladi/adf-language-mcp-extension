# Equation Input vs. Word-Problem Input

## First 10 records: smoke test

Provider/model: `groq` / `openai/gpt-oss-20b`; temperature: `0.7`.

| Metric | Equation-input agent | Word-problem agent |
| --- | ---: | ---: |
| Records | 10 | 10 |
| Correct | 10 | 8 |
| Accuracy | 100.0% | 80.0% |
| Wrong answer | 0 | 0 |
| Format failure | 0 | 1 |
| API error / timeout | 0 | 1 |
| Total wall-clock seconds | 6.906586 | 7.155956 |
| Average latency (ms) | 690.659 | 715.596 |
| LLM turns | 20 | 19 |
| MCP tool calls | 10 | 10 |
| MCP tool errors | 0 | 0 |
| No-tool answers | 0 | 0 |
| Maximum-step failures | 0 | 0 |
| Groq input tokens | 6060 | 6694 |
| Groq output tokens | 897 | 1735 |
| Groq total tokens | 6957 | 8429 |

The two runs used the same model, temperature, MCP tool set, and ordered problem IDs. The only intended input difference is equation text versus the original SVAMP word problem.

## Random 50 records (seed 42): Equation Input vs. Word-Problem Input

Provider/model: `groq` / `openai/gpt-oss-20b`; temperature: `0.7`.

| Metric | Equation-input agent | Word-problem agent |
| --- | ---: | ---: |
| Records | 50 | 50 |
| Correct | 14 | 0 |
| Accuracy | 28.0% | 0.0% |
| Wrong answer | 0 | 0 |
| Format failure | 0 | 0 |
| API error / timeout | 36 | 50 |
| Total wall-clock seconds | 18.878222 | 6.421922 |
| Average latency (ms) | 377.564 | 128.438 |
| LLM turns | 32 | 2 |
| MCP tool calls | 18 | 2 |
| MCP tool errors | 0 | 0 |
| No-tool answers | 0 | 0 |
| Maximum-step failures | 0 | 0 |
| Groq input tokens | 9821 | 666 |
| Groq output tokens | 2248 | 311 |
| Groq total tokens | 12069 | 977 |

The two runs used the same model, temperature, MCP tool set, and ordered problem IDs. The only intended input difference is equation text versus the original SVAMP word problem.

### Interpretation

This run is retained as an **execution-reliability record**, not a model-quality comparison. It contains 86 `api_error` outcomes out of 100 attempted model calls: 36 in the equation-input condition and all 50 in the word-problem condition. No arithmetic-tool error was recorded. Therefore, the observed 28% versus 0% accuracy cannot be attributed to the two input forms; the same seeded sample should be repeated when the provider completes both conditions reliably.
