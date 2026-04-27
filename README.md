# BARRED

<p align="center">
  <em>Custom Policy Guardrails via Asymmetric Debate</em>
</p>

<p align="center">
  <a href="https://discord.gg/YWbT87vAau"><img src="https://img.shields.io/badge/Join-Discord-5865F2?logo=discord&logoColor=white" alt="Join Discord"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-brightgreen.svg" alt="License: Apache 2.0"></a>
</p>

<p align="center">
  <a href="https://plurai.substack.com/">Newsletter</a> |
  <a href="BARRED.pdf">Paper</a> |
  <a href="https://plurai.ai/">Plurai</a>
</p>

---

This repository contains the test sets and evaluation code for the paper **[BARRED: Synthetic Training of Custom Policy Guardrails via Asymmetric Debate](BARRED.pdf)**.

It includes:
- **Test sets** for benchmarking policy guardrail models across multiple domains
- **Evaluation code** for running inference with LLM judges (e.g. Azure OpenAI, Google, Anthropic) or any SLM from HuggingFace

- [x] Benchmark
- [x] Inference
- [ ] Training

## About

BARRED is a benchmark for evaluating **custom policy guardrails** — classifiers that decide whether a given AI input or output complies with a stated, free-text policy rule. The benchmark spans **four guardrail tasks across three domains**: conversational policy enforcement, agentic output verification, and regulatory compliance.

Each sample consists of:
- a **policy rule** expressed in natural language (the predicate the guardrail must enforce),
- an **input** to be checked (a multi-turn dialogue, an agent-generated plan, or a Q&A pair), and
- a **ground-truth label** indicating whether the policy is violated.

The task for a guardrail model is to predict, given the rule and the input, whether the rule is satisfied or violated. All test sets in this repository are **human-curated** to ensure quality and scenario diversity.

| Dataset              | Domain | Input Type | Samples |
|----------------------|---|---|---------|
| `Message repetition` | Conversational policy enforcement | Multi-turn dialogue | 158 |
| `Privacy disclosure` | Conversational policy enforcement | Multi-turn dialogue | 112 |
| `Plan verification`  | Agentic output verification | Structured plan | 116 |
| `Health advice`      | Regulatory compliance (healthcare) | Q&A | 200 |

See [Datasets](#datasets) for a detailed description of each task.

## Links

- 📄 Read the [paper](BARRED.pdf).
- 🚀 Try BARRED-grade guardrails in production, powered by the same method benchmarked here, on the [Plurai platform](http://app.plurai.ai/).
- 🌐 Learn more about [Plurai](https://plurai.ai/).
- 💬 Join our [Discord community](https://discord.gg/YWbT87vAau).
- 🤗 Browse the datasets on [HuggingFace](https://huggingface.co/datasets/Plurai/BARRED).

## Setup

**Install dependencies:**
```bash
uv sync
```

**Configure environment:**

Copy `.env` and fill in your credentials:
```
AZURE_OPENAI__API_KEY=<your-api-key>
AZURE_OPENAI__ENDPOINT=https://<your-resource>.openai.azure.com
AZURE_OPENAI__API_VERSION=<api-version>
AZURE_OPENAI__DEPLOYMENT_NAME=<leave-it-empty>

GOOGLE__API_KEY=<your-api-key>

# Logging
LOG_LEVEL=INFO
LOG_TO_FILE=false
VERBOSE_CONSOLE_OUTPUT=false
LOGGING__FORMAT=console
```

## Running Inference

From the `src/` directory:
```bash
export PYTHONPATH=/path/to/BARRED/src
uv run --env-file ../.env python evaluations/evaluations_runner.py --test_config_file ../config/<config>
```

where `<config>` is one of:
- `test_config_message_repetition.yaml`
- `test_config_plan_verification.yaml`
- `test_config_healthe.yaml`
- `test_config_gps_disclosure.yaml`

## Configuration

Each config file has the following sections:

- **`classification_type`**: Either:
  - `input_block` - evaluate a single-piece block, e.g. a conversation transcript
  - `task_and_response` - evaluate whether a model response to a given task fulfills the policy

- **`dataset`**:
  - `name`: Dataset handler to use (`message_repetition`, `plan_verification`, `healthe`, `gps_disclosure`)
  - `test_file`: Path to the CSV test file  **or**
  - `hf_dataset`: HuggingFace dataset repo (e.g. `Plurai/BARRED`), with optional `hf_config` (defaults to `name`) and `hf_split` (defaults to `test`)
  - `cluster`: Free-text policy rule passed to the classifier as the rule to evaluate
  - `labels`: Possible output labels (e.g. `['True', 'False']` or `['PASS', 'FAIL']`)
  - `evaluator`: The model used for evaluation. Either:
    - `llm_config`: An LLM with `type`, `name`, `temperature`, and optional `model_kwargs`
    - `slm_model`: A fine-tuned SLM from HuggingFace (e.g. `unsloth/Qwen2.5-7B`)

- **`out_dir`**: Directory where results are saved.

---

## Datasets

### Message repetition

Inspired by [DynaGuard](https://arxiv.org/abs/2509.02563), a benchmark of multi-turn customer service dialogues, we constructed a test set around the rule: _"A user repeats or rephrases the same message 3 times"._ The classifier predicts whether this predicate holds on the input transcript. We significantly expanded the original 34 test cases with human-annotated samples, reaching a total of 158.

To ensure diversity, samples cover a wide range of scenarios: near-identical repetitions vs. subtle rephrasings, varying transcript lengths and conversation topics, different speakers making the repetition, and user messages with progressively broadening or narrowing intent compared to true repetitions.

**Columns:** `predicate`, `transcript`, `predicate_label`

### Privacy disclosure

Inspired by [DynaGuard](https://arxiv.org/abs/2509.02563), focusing on the predicate: _"The AI agent disclosed the GPS coordinate locations of individual employees"._ We extended the original 22 test cases with additional human-annotated samples to a total of 112.

To ensure diversity, samples cover a wide range of scenarios: exposure of precise locations in non-GPS formats, GPS coordinates embedded within structures such as URLs, accurate vs. coarse GPS coordinates, different levels of employee identifiability, and implicit disclosure by referencing information introduced earlier in the conversation.

**Columns:** `predicate`, `transcript`, `predicate_label`

### Plan verification

Inspired by the [GAIA](https://arxiv.org/abs/2311.12983) benchmark for General AI Assistants. We defined a guardrail task over LLM-generated research plans, where the classifier predicts whether a plan adheres to a set of instructions: the plan is allowed to use only the specified tools, it refers to them abstractly rather than as explicit calls, and ends with `\n<end_plan>`.

The original dataset contained a single failure mode: missing `<end_plan>` tag. We manually introduced diverse failure modes into valid plans to construct a balanced and varied set of non-adherent cases that cover the full space of possible violations.

**Columns:** `rule`, `task_input`, `original_task_output`, `violating_task_output`, `violation_type`

### Health advice

Based on the [HealthE](https://zenodo.org/records/7539391) benchmark (Gatto et al., 2023). Given a sentence or paragraph from a healthcare context, we generated a corresponding question to form a Q&A pair. The classifier predicts whether the agent's response to the presented question constitutes health advice. 200 samples were curated from the original dataset and processed to create the test set.

**Columns:** `predicate`, `transcript`, `predicate_label`

---

## HuggingFace Hub

The datasets are available at [Plurai/BARRED](https://huggingface.co/datasets/Plurai/BARRED).

**Load directly from the Hub:**
```python
from datasets import load_dataset
ds = load_dataset("Plurai/BARRED", "message_repetition", split="test")
```

Available configurations: `message_repetition`, `gps_disclosure`, `healthe`, `plan_verification`.

---

## Community & Support

- **From benchmark to production.** The [Plurai platform](http://app.plurai.ai/) ships the same method evaluated here — BARRED scores reflect production quality.
- Learn more about [Plurai](https://plurai.ai/).
- Join the discussion on [Discord](https://discord.gg/YWbT87vAau) to share results, request features, or get help running the benchmark.

---
