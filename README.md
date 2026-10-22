# ADF Language-to-MCP Extension

This repository is a small, reproducible extension of the Architectural Decision Framework (ADF) arithmetic benchmark by Garimella, Srivastava, and Sheth. It asks two additional questions:

> What changes when a model receives the original SVAMP word problem rather than the already-written arithmetic equation, and does that change across six input languages?

The project has no Beacon or health data. It uses the authors' 1,000 corrected SVAMP records only.

## Current reproducible pilot

The latest checked artifact is a matched 20-problem MCP-agent pilot using the complete five-model local agent list from the reused ADF configuration, temperature `0`, and seed `798`. It compares canonical equations with the same randomly selected word problems in English, Arabic, Chinese, French, Russian, and Spanish. See the [multi-model result table, agent diagnostics, limitations, and 35 raw runs](results/pilot_20_seed798_all_agent_models_T0/README.md).

This is a preliminary pilot, not the full 1,000-record multilingual evaluation. The translated inputs preserve all numeric literals but are machine-generated and still require human linguistic review.

The core English extension has also been run on all 1,000 records across all five local MCP-agent models. See the [validated full equation-versus-English report](results/full_1000_agent_equation_vs_english_T0/README.md).

## Study design

The study has two architectures and two input conditions. The four MCP tools—`add`, `subtract`, `multiply`, and `divide`—are available only to the agent architecture.

| Condition | Input | Purpose |
| --- | --- | --- |
| `author_equation` | The record's `Equation` field | Reuses the authors' equation-input condition. |
| `language_extension` | The record's `Body` and `Question` fields | Tests language interpretation, with or without the same MCP-tool workflow. |

`language_extension` can use the unchanged English input or one fixed translation in Arabic (`ar`), Simplified Chinese (`zh`), French (`fr`), Russian (`ru`), or Spanish (`es`). Together with English, these are the six official UN languages.

Every run is scored against the same record's exact `Answer` field. The extension reports answer accuracy, failure categories, latency, and provider usage. Agent runs additionally report LLM turns, MCP tool calls, tool errors, and maximum-step failures.

This is not a claim that all model families have identical capabilities. The upstream paper uses one local-model list for direct LLM runs and a different one for tool-using agents; this extension preserves that boundary.

## Model coverage from the reused ADF code

The earlier Groq example used only `openai/gpt-oss-20b`, so it did not show the complete upstream model set. `upstream_adf/code/config_llm.json` defines the following two model sweeps.

| Architecture | Models | Why this list |
| --- | --- | --- |
| Direct LLM, no tools | `llama3.2:1b`, `qwen2-math:1.5b`, `phi4-mini:3.8b`, `llama3.2:3b`, `granite4:3b` | These are the upstream `ollama_models`. |
| LLM + MCP tools | `llama3.2:1b`, `phi4-mini:3.8b`, `llama3.2:3b`, `qwen3.5:4b`, `granite4:3b` | These are the upstream `ollama_agent_models`. |
| Hosted model | `openai/gpt-oss-20b` through Groq | The single cloud configuration in the upstream code. |

`qwen2-math:1.5b` is direct-only because the upstream paper reports that it rejects tool calls. `qwen3.5:4b` is agent-only because the upstream paper did not include it in direct runs. The complete extension matrix therefore covers all seven distinct upstream model names in the architecture where the authors used them; it does not falsely label an unsupported model as a tool-calling agent.

## New contribution

The upstream ADF agent receives an already-written arithmetic equation. This extension adds a controlled word-problem condition in which the model instead receives the original SVAMP `Body` and `Question` text. It supports both the no-tool direct baseline and the MCP-agent workflow. For each matched architecture/model/temperature comparison, the scorer, record set, and—when applicable—the four arithmetic tools and maximum-step policy remain the same.

That design separates three questions:

1. Can the model use the MCP arithmetic tools when the equation is already available?
2. What additional execution failures appear when the model must infer the arithmetic from natural language before using those same tools?
3. Does the language-input result change when the same fixed word problem is presented in Arabic, Chinese, English, French, Russian, or Spanish?

`extension/language_agent.py` is the MCP-agent implementation. `extension/direct_language.py` implements the no-tool baseline. `extension/run_model_matrix.py` plans a complete and fair model/language sweep, and `extension/compare_multilingual.py` refuses to compare runs with different models, temperatures, tool sets, samples, or problem IDs.

## Provenance and attribution

`upstream_adf/` is an unchanged snapshot of the ADF repository's source code and corrected SVAMP data, supplied with this project. It is MIT licensed. Its original `LICENSE` and `CITATION.cff` files are retained unchanged. The upstream source is:

- Garimella, R., Srivastava, B., and Sheth, A. (2026). *Doing Less with More: A First-Principles Exploration of the Suitability of Agentic Computing over Alternative Architectural Choices*.
- Source repository: <https://github.com/Ritvik-G/adf>

The files in `extension/` and `tests/` are the new work in this repository. They reuse the upstream MCP server, MCP client, agent backends, strict answer scorer, and dataset. See [NOTICE.md](NOTICE.md) for the boundary between upstream and extension code.

## Layout

```text
upstream_adf/       unchanged, attributed ADF snapshot: code and data
extension/          word-problem agent loop and benchmark runner
data/multilingual_svamp/
                    fixed translations with review status recorded in metadata
tests/              offline provenance and agent-loop tests
results/runs/       local generated results; ignored by Git
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the hosted `openai/gpt-oss-20b` run, create one Groq key and set it privately in the terminal. Do not put a key in code or commit a `.env` file.

```bash
export GROQ_API_KEY='your-key-here'
```

The six local Ollama models need **no API keys**. Pull only the models you intend to run:

```bash
ollama pull llama3.2:1b
ollama pull qwen2-math:1.5b
ollama pull phi4-mini:3.8b
ollama pull llama3.2:3b
ollama pull granite4:3b
ollama pull qwen3.5:4b
```

The matrix runner reports missing models before it starts a run. It never downloads models automatically.

## Prepare the fixed multilingual corpus

Translation must happen once, before evaluation—not inside each model call. The repository includes an optional NLLB data-preparation script that preserves numeric literals, supports resuming after interruption, and records translation provenance.

```bash
pip install -r requirements-translation.txt
python extension/prepare_multilingual_svamp.py --languages ar zh fr ru es --resume
```

Read and spot-check the generated files before changing `metadata.review_status` from `machine-generated; human review required` to `reviewed`. The experiment loader rejects missing, incomplete, unreviewed, ID-mismatched, or number-changing translations. See [`data/multilingual_svamp/README.md`](data/multilingual_svamp/README.md) for the required file format.

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

### Direct baseline and full model/language matrix

The no-tool baseline keeps the upstream direct-only model separate from the MCP-agent models:

```bash
python extension/run_direct_experiment.py --condition language_extension \
  --provider ollama --model qwen2-math:1.5b --temperature 0 --limit 10
```

Plan the full experiment before running it. Planning is the default: it does not call a model, download a model, or use an API key.

```bash
python extension/run_model_matrix.py \
  --architecture all --provider all --languages all --limit 1000 \
  --write-plan results/full_matrix_plan.json
```

The full plan contains 154 jobs: direct and MCP-agent architectures, their correct upstream model lists, canonical-equation baselines, six word-problem languages, and the upstream temperature settings. It reports missing translations, local models, or MCP server before any run starts.

After those checks pass, start the full matrix explicitly and preserve completed work after interruption:

```bash
python extension/run_model_matrix.py \
  --architecture all --provider all --languages all --limit 1000 \
  --write-plan results/full_matrix_plan.json --execute --resume
```

Compare six matching word-problem runs with:

```bash
python extension/compare_multilingual.py \
  results/matrix_runs/<six-matching-word-language-files>.json \
  --output results/multilingual_comparison.md
```

Run offline checks at any time:

```bash
python -m unittest discover -s tests -v
```

## Interpretation boundary

The deterministic tools perform the arithmetic. A tool-agent failure can come from understanding the word problem, selecting or sequencing tools, formatting a final answer, or provider failure. A direct-model failure may also be arithmetic reasoning itself. A translation-induced error is a separate data-quality risk, which is why translations are fixed, validated, and reviewed before evaluation.

Do not compare a result from this repository directly with a saved upstream paper number unless the same model, temperature, environment, and scoring rule were used. The key comparisons are within a matched model/temperature: equation input versus word-problem input, and one fixed word problem language versus another.

## Included test evidence

[`results/comparison.md`](results/comparison.md) contains the initial 10-record local Groq smoke test. It is included only to show the expected output format and tool-use measurements; it is not a final evaluation.
