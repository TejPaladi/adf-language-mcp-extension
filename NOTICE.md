# Third-party notice and authorship boundary

## Upstream ADF code and data

The `upstream_adf/` directory is an unmodified snapshot of the Architectural Decision Framework (ADF) repository by Ritvik Garimella, Biplav Srivastava, and Amit Sheth:

<https://github.com/Ritvik-G/adf>

It includes the upstream MIT `LICENSE` and `CITATION.cff` files. Its code and data are not claimed as original work in this repository.

The extension specifically reuses these upstream components:

- `code/mcp_server.py`: the four arithmetic MCP tools;
- `code/mcp_client.py`: the persistent MCP client session;
- `code/mcp_agent.py`: provider adapters, tool-schema conversion, and the equation-input agent loop;
- `code/main.py`: strict numeric answer classification; and
- `data/train.json` and `data/test.json`: the corrected ADF/SVAMP records.

## New extension work

The following directories contain the project-specific addition:

- `extension/`: a word-problem MCP-agent loop and a runner that compares it with the upstream equation-input agent condition;
- `tests/`: checks for dataset provenance and the extension loop; and
- root documentation and Git configuration.

The new code does not modify the upstream implementation. It changes only the user input supplied to an otherwise equivalent MCP-agent loop: `Body + Question` instead of `Equation`.

## Citation

Any report or GitHub README based on this project should cite the ADF paper and repository, not only this extension repository.
