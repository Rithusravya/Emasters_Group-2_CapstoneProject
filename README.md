# CodeGen

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A capstone project (eMasters CS: AI & ML, Group 2) that fine-tunes a small open-source code language model — **Qwen2.5-Coder-0.5B-Instruct** — with **LoRA**, builds a **hybrid semantic + structural (AST/MQL) retrieval (RAG)** layer on top of it, and evaluates the result against the un-adapted base model, across **natural-language-to-MongoDB-query (Text-to-MongoDB)** generation, plus supporting documentation-generation, program-synthesis, and commit-message-generation tasks.

> **Note on scope:** the project's primary evaluated task is **Text-to-MongoDB**, generated from the Spider text-to-SQL benchmark's natural-language questions. Documentation, program-synthesis, and commit-message generators are also implemented and share the same generation/evaluation machinery, but are exercised on small hand-authored example sets rather than a benchmark.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Key Capabilities](#key-capabilities)
- [System Architecture](#system-architecture)
- [Pipeline Stages](#pipeline-stages)
- [Benchmark Results](#benchmark-results)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Layout](#repository-layout)
- [Configuration](#configuration)
- [Requirements](#requirements)
- [Security Note](#security-note)
- [License](#license)

---

## Executive Summary

CodeGen combines **Retrieval-Augmented Generation (RAG)** and **parameter-efficient LoRA fine-tuning** on top of a compact instruction-tuned code model to translate natural-language questions into **MongoDB Query Language (MQL)** statements. The evaluation dataset is derived from the **Spider** cross-domain text-to-SQL benchmark (up to 1,034 dev-split questions), whose gold SQL has been converted to equivalent MongoDB queries and is shipped pre-generated in the repository (`qwen_spider_mongodb_conversion.json`, `spider_real_execution_mongo.json`) — no separate download is required to run the baseline/fine-tuned evaluation notebooks.

Retrieval embeddings are produced by an external sentence-embedding model (`BAAI/bge-m3` by default), and are combined with a structural fingerprint index — extracting MQL operations, aggregation stages, and filter operators (or SQL/Python constructs, depending on configuration) — so retrieval can be reranked on structure, not just vector similarity.

---

## Key Capabilities

| Capability | Description |
|---|---|
| **Text-to-MongoDB Generation** | Natural-language question → MongoDB Query Language (MQL) statement, plus documentation, program-synthesis, and commit-message generation sharing the same pipeline |
| **External Embeddings** | Retrieval embeddings from a dedicated Hugging Face embedding model (`BAAI/bge-m3` by default, `BAAI/bge-small-en-v1.5` config-selectable) via `HFEmbedder` |
| **Hybrid Retrieval** | FAISS (`IndexFlatIP`) dense semantic search fused with an inverted-index structural/MQL feature reranker (default: 0.7 semantic / 0.3 structural) |
| **LoRA Fine-Tuning** | PEFT LoRA adapter (r=16, α=32) on the attention + MLP projections, with a custom training loop (gradient accumulation, prompt-token label masking, MPS-cache management) |
| **Optional LLM Baseline** | Pluggable Gemini-API baseline generator (`google-genai`) for comparing the small fine-tuned LM against a large hosted model |
| **Evaluation Suite** | BLEU, ROUGE-1/L, CodeBERTScore (via `codebert-base`), token-overlap Response Accuracy, RAG gain-over-baseline, top-K sweep |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Configuration & Data                                         │
│  • Load config.yaml (paths, model, LoRA, retrieval, eval)       │
│  • Load pre-generated Spider→MongoDB question/query pairs       │
│  • (Optional) convert raw Spider SQL → MongoDB via LLM          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ 2. Model Loading & LoRA Fine-Tuning                             │
│  • Load Qwen2.5-Coder-0.5B-Instruct (tokenizer + base weights)  │
│  • Attach LoRA adapter, train on Question → MongoDB Query pairs │
│  • Save / reload adapter checkpoint                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────────┐
│ 3. Generation Tasks                                                │
│  • Text-to-MongoDB (primary, benchmarked)                          │
│  • Documentation, Program Synthesis, Commit Messages (illustrative)│
│  • Base model vs. LoRA-fine-tuned model                            │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ 4. Hybrid Indexing & RAG                                        │
│  • HFEmbedder → FAISS semantic index (IndexFlatIP)              │
│  • ASTIndex → structural fingerprints (MQL / SQL / Python)      │
│  • HybridRetriever → normalized semantic + structural blend     │
│  • RAGPipeline → context injection + MongoDB-query generation   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│ 5. Evaluation & Comparison                                        │
│  • Base vs. LoRA vs. RAG (ModelComparator)                        │
│  • RAG gain-over-baseline & top-K sweep (RAGEvaluator)            │
│  • Charts (matplotlib/seaborn) + JSON metrics persisted to output │
└───────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

The pipeline is run through a set of Jupyter notebooks rather than a single CLI entry point:

| Notebook | Stage |
|---|---|
| `baseline_model.ipynb` | Loads the **un-adapted base model**, runs Text-to-MongoDB generation over the Spider-derived question set, saves metrics to `output/generated/baseline_model_results.json` and predictions to `outputs/generated/baseline_mongo_predictions.json` |
| `finetuned_model.ipynb` | Loads/trains the **LoRA adapter**, re-runs Text-to-MongoDB generation, saves metrics to `output/generated/finetuned_model_results.json` and predictions to `outputs/generated/finetuned_mongo_predictions.json`, and produces a baseline-vs-fine-tuned comparison |
| `spider_evaluation.ipynb` | SQL execution-accuracy evaluation against the raw Spider SQLite databases, plus the SQL→MongoDB conversion step used to produce `qwen_spider_mongodb_conversion.json` |
| `main.ipynb` | Colab-oriented end-to-end pipeline (setup, indexing, RAG, comparison) |

Supporting production code lives in `src/` (see [Repository Layout](#repository-layout)) and is imported by the notebooks — it is not currently exposed as a standalone CLI script.

### Task-Specific Generators (`src/generators/`)

| Task | Module | Data used | Primary Metrics |
|---|---|---|---|
| **Text-to-MongoDB** | `text_to_mongo_generator.py` | Spider-derived question/MongoDB-query pairs (`qwen_spider_mongodb_conversion.json`, up to 1,034 questions; a 100-question subset is used for the reported metrics) | BLEU, ROUGE-1/L, CodeBERTScore, Response Accuracy |
| **Program Synthesis** | `program_generator.py` (also the shared `GenerationPipeline`) | Hand-authored illustrative prompts | Qualitative + shared metric suite |
| **Documentation Generation** | `doc_generator.py` | 5-shot prompting over hand-authored examples | Qualitative + shared metric suite |
| **Commit Message Generation** | `commit_generator.py` | Hand-authored git-diff examples | Qualitative + shared metric suite |
| **LLM Baseline (optional)** | `llm_baseline.py` | Google Gemini API (`google-genai`) | Comparative scoring against the small LM |

### Hybrid Indexing & Retrieval (`src/vectorDB/`)

| Component | Technology | Purpose |
|---|---|---|
| `embedder.py` (`HFEmbedder`) | `BAAI/bge-m3` (default) via `AutoModel`/`AutoTokenizer` | Dense retrieval embeddings from an external Hugging Face model |
| `semantic_index.py` (`SemanticIndex`) | FAISS `IndexFlatIP` | Dense semantic vector store (cosine similarity via normalized inner product); saved as `semantic_index.faiss` + `semantic_index.metadata.pkl` |
| `ast_index.py` (`ASTIndex`) | Inverted-index feature extractor | Structural fingerprints: MongoDB collections/operations/pipeline stages/filter operators when `language="mongodb"`, with SQL and Python extractors also available |
| `hybrid_retriever.py` (`HybridRetriever`) | Score-normalized blend | Combines min-max-normalized semantic and structural scores using `semantic_weight` / `ast_weight` (default 0.7 / 0.3) |
| `index_manager.py` (`IndexManager`) | Orchestrator | Builds, saves, loads, and auto-detects the semantic + AST indices as a unit and exposes a single `search()` |

### RAG Pipeline (`src/rag/rag_pipeline.py`)

`RAGPipeline` retrieves top-K context examples via the hybrid retriever, builds a MongoDB-query-generation prompt from those examples (`build_prompt`), generates a completion, and post-processes the output with `_clean_mongo_output` to isolate a single valid `db.<collection>....` statement.

### Evaluation & Persistence (`src/evaluation/`)

| Component | Function |
|---|---|
| `metrics.py` | BLEU, ROUGE, CodeBERTScore, and a token-overlap "Response Accuracy" metric with an MQL-aware tokenizer |
| `comparator.py` (`ModelComparator`) | Base vs. LoRA vs. RAG side-by-side comparison across BLEU/ROUGE/CodeBERTScore/Response Accuracy |
| `rag_evaluation.py` (`RAGEvaluator`) | RAG gain-over-baseline measurement and top-K sweep (K ∈ {1, 2, 3, 5, 8} by default), with a formatted sweep table |
| `visualization.py` (`ResultVisualizer`) | Seaborn/matplotlib bar-chart generation, saved to `output/plots/` |

---

## Benchmark Results

Numbers below are taken directly from `output/generated/baseline_model_results.json`, `output/generated/finetuned_model_results.json`, and `output/comparison_metrics.json`.

### Baseline (un-adapted) vs. LoRA Fine-Tuned — Text-to-MongoDB (100-question subset)

| Metric | Base Model | LoRA Fine-Tuned |
|---|:---:|:---:|
| BLEU | 0.0008 | **0.2019** |
| CodeBERTScore | 0.8004 | **0.8817** |
| ROUGE-1 | 0.2504 | **0.5360** |
| ROUGE-L | 0.2263 | **0.4462** |
| Response Accuracy | 0.3297 | **0.7624** |
| Avg. latency (ms) | 38,890 | 54,060 |

LoRA fine-tuning improves every metric substantially — Response Accuracy roughly doubles and BLEU moves from near-zero to 0.20 — at the cost of higher per-query latency.

### Base vs. LoRA vs. RAG (5-question comparison slice, `output/comparison_metrics.json`)

| Metric | Base Model | LoRA Model | RAG Pipeline |
|---|:---:|:---:|:---:|
| BLEU | 0.2259 | 0.1654 | 0.0044 |
| ROUGE-1 | 0.5598 | 0.5048 | 0.2413 |
| ROUGE-L | 0.4560 | 0.4199 | 0.2329 |
| BERTScore | 0.8759 | 0.8650 | 0.7694 |
| Response Accuracy | 0.9520 | 0.9520 | 0.6293 |
| Avg. latency (ms) | 20,406 | 25,387 | 33,931 |

- On this small 5-question evaluation slice, the **RAG-augmented pipeline underperformed both the base and LoRA models** on every metric, while also adding the most latency — a genuine, reproducible negative result on this slice, not a placeholder.
- The best top-K found by the accompanying sweep on this slice is **K = 2** (`best_k` in `comparison_metrics.json`); larger K did not recover the gap.
- Because this comparison is over only 5 questions, treat it as indicative rather than statistically robust — the 100-question base-vs-LoRA numbers above are the more reliable signal.

Plots are saved under `output/plots/` (`model_comparison_full.png`, `radar_comparison.png`, `topk_gain_sweep.png`) and `src/plots/` (`text_to_mongo_baseline_metrics.png`, `text_to_mongo_finetuned_metrics.png`, `baseline_vs_finetuned_text_to_mongo.png`, `baseline_vs_finetuned_latency.png`).

---

## Dataset

This project's Text-to-MongoDB task is derived from the **[Spider](https://yale-lily.github.io/spider)** dataset — a large-scale, cross-domain text-to-SQL benchmark.

- **Pre-generated data (used by `baseline_model.ipynb` / `finetuned_model.ipynb`, no download needed):**
  - `qwen_spider_mongodb_conversion.json` — Spider natural-language questions paired with LLM-generated MongoDB queries (1,114 records), used for LoRA training and Text-to-MongoDB evaluation.
  - `spider_real_execution_mongo.json` — the full 1,034-question Spider dev split with gold SQL, used as the source for the conversion above and for SQL-execution baselines.
  - Both files are committed at the repository root and loaded automatically by `src.data.data_loader.DatasetLoader`.

- **Raw Spider data (only needed for `spider_evaluation.ipynb`'s SQL execution-accuracy step and to regenerate the conversion files):**
  1. Download the dataset from Google Drive: **[Spider dataset (Google Drive)](https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view)**, or see the [official Spider dataset page](https://yale-lily.github.io/spider) for the source release and licensing terms.
  2. Unzip and place the contents under `data/raw/Spider/spider_data/`, so that `dev.json`, `tables.json`, and the per-database SQLite files sit alongside each other, e.g.:

     ```
     data/raw/Spider/spider_data/
     ├── dev.json
     ├── tables.json
     └── database/
         ├── concert_singer/
         │   └── concert_singer.sqlite
         └── ...
     ```

  3. Run the SQL execution and SQL→MongoDB conversion cells in `spider_evaluation.ipynb`.

`data/raw/` and `data/processed/` are excluded via `.gitignore` since they're large/downloaded rather than versioned; `data/indices/` (the built FAISS + AST indices) **is** committed in this snapshot.

---

## Installation

### Prerequisites

```bash
Python 3.10 or higher
A CUDA-capable GPU is optional — the code auto-detects CUDA > MPS (Apple Silicon) > CPU
Recommended: 16 GB RAM, 8 GB+ VRAM if using GPU
```

### Step 1: Clone Repository

```bash
git clone https://github.com/Rithusravya/Emasters_Group-2_CapstoneProject.git
cd Emasters_Group-2_CapstoneProject
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
# OR
venv\Scripts\activate         # Windows
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## Quick Start

This project is driven through notebooks, not a CLI. Typical order:

1. **Baseline evaluation** — open `baseline_model.ipynb` and run all cells to evaluate the un-adapted base model on Text-to-MongoDB generation using the pre-generated dataset (no download required).
2. **LoRA fine-tuning + evaluation** — open `finetuned_model.ipynb` to train (or reload) the LoRA adapter at `models/checkpoints/lora_finetuned/`, re-run the same evaluation, and generate a baseline-vs-fine-tuned comparison.
3. **RAG / hybrid retrieval** — build or reuse the indices in `data/indices/` via `IndexManager`, then use `RAGPipeline` to compare RAG-augmented generation against the LoRA-only baseline (see `main.ipynb`).
4. **Spider SQL execution evaluation (optional)** — after downloading the raw Spider dataset as described in [Dataset](#dataset), open `spider_evaluation.ipynb` for SQL-execution-based scoring and to regenerate the SQL→MongoDB conversion files.

Programmatic usage of the core pipeline, e.g. for Text-to-MongoDB generation:

```python
from src.config import load_config
from src.models.load_model import ModelLoader
from src.generators.program_generator import GenerationPipeline
from src.generators.text_to_mongo_generator import TextToMongoGenerator

config = load_config()  # reads configs/config.yaml
loader = ModelLoader(config)
models, tokenizer = loader.load_models(lora_path=config.lora.output_dir)

pipeline = GenerationPipeline(models["lora"] or models["base"], tokenizer, config.generation)
mongo_gen = TextToMongoGenerator(pipeline)

result = mongo_gen.generate_single(
    question="How many singers do we have?",
    schema="collection: singer { singer_id, name, age }",
)
print(result["generated_query"])   # e.g. db.singer.countDocuments()
```

---

## Repository Layout

```
Emasters_Group-2_CapstoneProject/
├── configs/
│   └── config.yaml                    # Project, model, LoRA, retrieval, eval & output paths
├── data/
│   └── indices/                       # Committed FAISS index, AST store, FAISS metadata
├── models/
│   ├── checkpoints/lora_finetuned/    # Saved LoRA adapter + tokenizer files
│   └── codebert-base/                 # Local CodeBERT checkpoint used for CodeBERTScore
├── src/
│   ├── config.py                      # YAML → dataclass config loader
│   ├── data/
│   │   └── data_loader.py             # Loads/normalizes the Spider→MongoDB question/query pairs
│   ├── models/
│   │   └── load_model.py              # Base model + LoRA loading, training loop, adapter save/load
│   ├── generators/
│   │   ├── program_generator.py       # Core GenerationPipeline (shared by all tasks)
│   │   ├── text_to_mongo_generator.py # Text-to-MongoDB generator (primary task)
│   │   ├── doc_generator.py           # Documentation generator (5-shot)
│   │   ├── commit_generator.py        # Commit-message generator
│   │   └── llm_baseline.py            # Optional Gemini-API LLM baseline
│   ├── vectorDB/
│   │   ├── embedder.py                # HFEmbedder — external embedding-model backend (BGE-M3)
│   │   ├── semantic_index.py          # FAISS semantic index manager
│   │   ├── ast_index.py               # MongoDB/SQL/Python structural fingerprint indexer
│   │   ├── hybrid_retriever.py        # Normalized semantic + structural blended retrieval
│   │   └── index_manager.py           # Builds/saves/loads/auto-detects the index pair
│   ├── rag/
│   │   └── rag_pipeline.py            # Context retrieval, MongoDB-prompt building, RAG generation
│   ├── evaluation/
│   │   ├── metrics.py                 # BLEU, ROUGE, CodeBERTScore, Response Accuracy
│   │   ├── comparator.py              # Base vs. LoRA vs. RAG comparison
│   │   ├── rag_evaluation.py          # RAG gain-over-baseline + top-K sweep
│   │   └── visualization.py           # Chart generation
│   └── plots/                         # Saved comparison charts (baseline/fine-tuned/latency)
├── output/
│   ├── generated/                     # baseline_model_results.json, finetuned_model_results.json
│   ├── plots/                         # model_comparison_full.png, radar_comparison.png, topk_gain_sweep.png
│   └── comparison_metrics.json        # Base vs. LoRA vs. RAG comparison (5-question slice)
├── outputs/generated/                 # baseline_mongo_predictions.json, finetuned_mongo_predictions.json
├── qwen_spider_mongodb_conversion.json# Spider questions + LLM-generated MongoDB queries (1,114 records)
├── spider_real_execution_mongo.json   # Full Spider dev split (1,034) with gold SQL
├── baseline_model.ipynb               # Base model Text-to-MongoDB evaluation
├── finetuned_model.ipynb              # LoRA fine-tuning + evaluation + comparison
├── spider_evaluation.ipynb            # SQL execution evaluation + SQL→MongoDB conversion
├── main.ipynb                         # Colab end-to-end pipeline entry point
├── requirements.txt
└── README.md
```

---

## Configuration (`configs/config.yaml`)

Key fields actually read by `src/config.py`:

```yaml
model_name: "Qwen/Qwen2.5-Coder-0.5B-Instruct"
lora_adapter_path: "models/checkpoints/lora_finetuned"

embedding_model: "BAAI/bge-small-en-v1.5"   # used only when embedding.source == "external"
embedding:
  source: "external"          # 'code_lm' (legacy, unused) or 'external' (default)
  external_model: "BAAI/bge-small-en-v1.5"
  max_length: 512
  batch_size: 32

retrieval:
  hybrid: true
  structural_weight_alpha: 0.7
  structural_language: "mongodb"   # "mongodb", "sql", or "python"
  topk_sweep: [1, 2, 3, 5, 8]

llm_baseline:
  backend: "api"               # "local" or "api" (Gemini)
  local_model: "Qwen/Qwen2.5-Coder-0.5B-Instruct"
  api_model: "gemini-3.6-flash"
  api_key_env: "<name of the environment variable holding your Gemini API key>"
  max_new_tokens: 256
  temperature: 0.2

data_paths:
  raw_dir: "data/raw"
  spider: "spider_real_execution_mongo.json"

indices_paths:
  faiss_index: "data/indices/faiss_semantic.index"
  ast_store: "data/indices/ast_store.pkl"

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  batch_size: 1
  learning_rate: 0.0002
  epochs: 1
  output_dir: "models/checkpoints/lora_finetuned"
  trainable_fraction: 1.0

indexing:
  top_k: 3
  vector_dim: 1024            # must match the embedding model's output dimension

outputs:
  logs_dir: "output/logs"
  metrics_path: "output/metrics/evaluation_metrics.json"
  plots_dir: "output/plots"
  generated_dir: "output/generated"
```

---

## Requirements

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.5.1 | Core deep learning framework |
| `transformers` | 4.46.3 | Pretrained model + tokenizer loading (Qwen2.5-Coder) |
| `peft` | 0.13.2 | LoRA fine-tuning |
| `datasets` | 3.1.0 | Dataset utilities |
| `faiss-cpu` | 1.9.0.post1 | Dense vector similarity search |
| `sentence-transformers` | 3.3.1 | Embedding-model utilities |
| `huggingface-hub` | 0.26.2 | Model/checkpoint downloads |
| `tree-sitter` / `tree-sitter-python` | 0.23.2 / 0.23.6 | (Available for AST-based parsing) |
| `codebleu` | 0.7.0 | CodeBLEU metric |
| `codebert-score` | 0.3.13 | CodeBERTScore semantic similarity metric |
| `evaluate` | 0.4.3 | Hugging Face evaluation metrics |
| `rouge-score` | 0.1.2 | ROUGE metrics |
| `nltk` | 3.9.1 | Text preprocessing / BLEU support |
| `PyYAML` | 6.0.2 | Config parsing |
| `matplotlib` / `seaborn` | 3.10.0 / 0.13.2 | Plotting |
| `numpy` | 2.1.3 | Numerical computing |
| `google-genai` | 1.0.0 | Gemini API client for the optional LLM baseline |

See `requirements.txt` for the full pinned list.

---
---

## Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.

---

## Citation

```bibtex
@software{codegen_capstone_2026,
  title  = {CodeGen: Small Code-LM Fine-Tuning and Hybrid RAG Pipeline for Text-to-MongoDB},
  author = {Emasters Group 2},
  year   = {2026},
  url    = {https://github.com/Rithusravya/Emasters_Group-2_CapstoneProject}
}
```

---

<div align="center">

**Built by IIIT-H eMasters Group 2**

[⬆ Back to Top](#codegen-hybrid-rag-pipeline-for-text-to-mongodb)

</div>
