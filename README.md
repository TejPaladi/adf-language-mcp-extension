# ADF Language-to-MCP Extension

This repository is a small, reproducible extension of the Architectural Decision Framework (ADF) arithmetic benchmark by Garimella, Srivastava, and Sheth. It asks one additional question:

> What changes when an MCP agent receives the original SVAMP word problem rather than the already-written arithmetic equation?

The project has no Beacon or health data. It uses the authors' 1,000 corrected SVAMP records only.

## Study design

Both agent conditions use the same four deterministic MCP tools: `add`, `subtract`, `multiply`, and `divide`.

| Condition | Input to the agent | Purpose |
| --- | --- | --- |
| `author_equation` | The record's `Equation` field | Reuses the authors' MCP-agent condition. |
| `language_extension` | The record's `Body` and `Question` fields | Tests language interpretation plus the same MCP-tool workflow. |

Every run is scored against the same record's exact `Answer` field. The extension reports answer accuracy, failure categories, latency, LLM turns, MCP tool calls, tool errors, and provider-reported cost information when available.

This is **not** a reproduction of every ADF method. The vendored upstream code remains available for its direct-function, REST, SOAP, MCP, and direct-LLM conditions. This repository focuses on the controlled comparison between equation input and word-problem input for the MCP-agent path.

## New contribution

The upstream ADF agent receives an already-written arithmetic equation. This extension adds one controlled condition in which the agent instead receives the original SVAMP `Body` and `Question` text. The MCP server, four arithmetic tools, model provider, strict numeric scorer, record set, and maximum-step policy remain the same.

That design separates two questions:

1. Can the model use the MCP arithmetic tools when the equation is already available?
2. What additional execution failures appear when the model must infer the arithmetic from natural language before using those same tools?

`extension/language_agent.py` is the project-specific implementation. `extension/run_experiment.py` saves per-record tool-use traces and `extension/compare_results.py` refuses to compare runs with different models, temperatures, tool sets, samples, or problem IDs.

## Provenance and attribution

`upstream_adf/` is an unchanged snapshot of the ADF repository's source code and corrected SVAMP data, supplied with this project. It is MIT licensed. Its original `LICENSE` and `CITATION.cff` files are retained unchanged. The upstream source is:

- Garimella, R., Srivastava, B., and Sheth, A. (2026). *Doing Less with More: A First-Principles Exploration of the Suitability of Agentic Computing over Alternative Architectural Choices*.
- Source repository: <https://github.com/Ritvik-G/adf>

The files in `extension/` and `tests/` are the new work in this repository. They reuse the upstream MCP server, MCP client, agent backends, strict answer scorer, and dataset. See [NOTICE.md](NOTICE.md) for the boundary between upstream and extension code.

## Layout

```text
upstream_adf/       unchanged, attributed ADF snapshot: code and data
extension/          word-problem agent loop and benchmark runner
tests/              offline provenance and agent-loop tests
results/runs/       local generated results; ignored by Git
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a Groq run, set the key privately in the terminal. Do not put a key in code or commit a `.env` file.

```bash
export GROQ_API_KEY='your-key-here'
```

For a local-model run, install Ollama and pull one of the models named in `upstream_adf/code/config_llm.json`.

## Run

Start the reused MCP calculator server in one terminal:

```bash
source .venv/bin/activate
cd upstream_adf/code
python mcp_server.py
```

From the repository root, run a small smoke test first:

```bash
python extension/run_experiment.py --condition author_equation --provider groq --limit 10
python extension/run_experiment.py --condition language_extension --provider groq --limit 10
```

Then use the same provider, model configuration, temperature, and all 1,000 records for each condition:

```bash
python extension/run_experiment.py --condition author_equation --provider groq
python extension/run_experiment.py --condition language_extension --provider groq
```

Each command prints a short summary and saves a JSON file in `results/runs/`. Compare the two matching saved files with:

```bash
python extension/compare_results.py \
  results/runs/<equation-run>.json \
  results/runs/<word-problem-run>.json \
  --output results/comparison.md
```

The comparison command refuses to combine runs that used different models, temperatures, MCP tools, or problem IDs.

The Groq model and endpoint come from the unchanged upstream configuration. For an Ollama run, specify a locally installed tool-calling model:

```bash
python extension/run_experiment.py --condition language_extension \\
  --provider ollama --model granite4:3b --temperature 0
```

Run offline checks at any time:

```bash
python -m unittest discover -s tests -v
```

## Interpretation boundary

The deterministic tools perform the arithmetic. A language-model failure in this study can come from understanding the word problem, selecting or sequencing tools, formatting a final answer, or provider failure. It is not evidence that the arithmetic tools failed.

Do not compare a result from this repository directly with a saved upstream paper number unless the same model, temperature, environment, and scoring rule were used. The point of the extension is the controlled within-run comparison: equation input versus natural-language input under the same MCP-tool architecture.

## Included test evidence

[`results/comparison.md`](results/comparison.md) contains the initial 10-record local Groq smoke test. It is included only to show the expected output format and tool-use measurements; it is not a final evaluation.
