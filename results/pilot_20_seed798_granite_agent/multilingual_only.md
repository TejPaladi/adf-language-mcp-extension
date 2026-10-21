# Multilingual word-problem comparison

Architecture: `mcp_agent`; provider/model: `ollama` / `granite4:3b`; temperature: `0.0`.

| Language | Correct | Accuracy | Wrong | Format | API error | Avg latency (ms) | Tool calls | LLM turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arabic (`ar`) | 9 | 45.0% | 11 | 0 | 0 | 2509.942 | 22 | 42 |
| English (`en`) | 17 | 85.0% | 3 | 0 | 0 | 2264.78 | 24 | 44 |
| Spanish (`es`) | 11 | 55.0% | 8 | 1 | 0 | 2390.641 | 25 | 45 |
| French (`fr`) | 8 | 40.0% | 11 | 1 | 0 | 2298.26 | 23 | 43 |
| Russian (`ru`) | 10 | 50.0% | 10 | 0 | 0 | 2336.933 | 22 | 42 |
| Chinese (Simplified) (`zh`) | 8 | 40.0% | 11 | 1 | 0 | 2325.083 | 24 | 44 |

All rows use the same model, temperature, architecture, tool set, scoring rule, and ordered problem IDs. Only the fixed language of the SVAMP body and question differs.
