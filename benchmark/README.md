# Benchmark Dataset

## Structure

```
benchmark/
  human/      # Human-written texts
  llm/        # LLM-generated texts
  run_benchmark.py
```

## File Format

Each text file begins with a metadata header, then `---`, then the raw text body.

```
source: human|llm
lang: en|tr
model: chatgpt|claude|gemini  (only for llm/)
---
<text>
```

## Running

```bash
cd ~/Desktop/writing-analyzer
source .venv/bin/activate
python benchmark/run_benchmark.py
```

## Adding New Samples

Drop `.txt` files into `human/` or `llm/` using the format above.
The runner auto-discovers all `.txt` files in both directories.
