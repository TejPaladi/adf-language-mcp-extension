# Matched 20-problem comparison across ADF MCP-agent models

## Controlled setup

Each cell uses the same 20 randomly selected SVAMP IDs (seed `798`), temperature `0`, four MCP arithmetic tools, strict ADF numeric scorer, and maximum-step policy. Only the model and input representation/language change.

The five models are the complete local `ollama_agent_models` list from the reused ADF configuration. `qwen2-math:1.5b` is not in this table because the ADF project records it as direct-only: it does not support tool calling.

## Strict answer accuracy

| Model | Equation | English | Arabic | Chinese | French | Russian | Spanish | Word-language mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama3.2:1b` | 0% | 5% | 10% | 0% | 0% | 0% | 5% | 3.3% |
| `llama3.2:3b` | 20% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| `phi4-mini:3.8b` | 75% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| `qwen3.5:4b` | 100% | 80% | 35% | 15% | 30% | 35% | 55% | 41.7% |
| `granite4:3b` | 100% | 85% | 45% | 40% | 40% | 50% | 55% | 52.5% |

## Agent diagnostics across all 140 attempts per model

Tool-use compliance is the share of attempts that made at least one actual MCP tool call. A tool error is counted separately and can occur after a model attempted a call.

| Model | Correct | Tool-use compliance | Format failures | Wrong answers | API/step failures | Tool errors | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama3.2:1b` | 4/140 | 97.9% | 96 | 40 | 0 | 43 | 2,734 ms |
| `llama3.2:3b` | 4/140 | 92.9% | 116 | 19 | 1 | 6 | 2,348 ms |
| `phi4-mini:3.8b` | 15/140 | 0% | 112 | 13 | 0 | 0 | 2,036 ms |
| `qwen3.5:4b` | 70/140 | 51.4% | 69 | 1 | 0 | 0 | 9,977 ms |
| `granite4:3b` | 83/140 | 100% | 3 | 54 | 0 | 0 | 1,587 ms |

## What the pilot shows

- Granite and Qwen3.5 both reached 100% on equation input. Granite had the strongest mean across the six word-problem languages (52.5%), followed by Qwen3.5 (41.7%).
- Accuracy alone does not describe agent quality. Granite used MCP tools in every attempt with no tool errors. Phi4-mini made no actual tool calls in any of its 140 attempts. Llama models often called tools but frequently violated the final number-only response contract.
- A format failure is not automatically wrong arithmetic. Examples include `ANSWER: 70.0`, a sentence containing the result, or a JSON-looking tool request emitted as plain text rather than as a tool call. The strict scorer rejects all of these, consistently with the reused ADF protocol.

## Limits

This is a preliminary model-behavior pilot, not a language ranking. It uses only 20 problems, and the five non-English corpora are machine-generated NLLB translations whose numeric literals were validated but whose linguistic quality has not yet been human-reviewed. Model capability and translation quality are therefore confounded. The hosted Groq model was not included; this table deliberately compares the local MCP-agent models from the ADF configuration without requiring an API key.

All 35 raw JSON files in this directory contain the individual outcomes and tool-use traces.
