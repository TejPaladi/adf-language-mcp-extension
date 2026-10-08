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
