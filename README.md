# Teaching a small model to reason in Portuguese: a PEFT study on Lumi-4B

**Authors:** @gustavobarrosss · @Pires-De-Andrade

> A study project by undergraduate students, still in progress.
> Status as of 2026-11-08. **The results below are preliminary**: they come from the validation sets, and none of our trained models has gone through the final test suite yet.

## Abstract

Language models that "think before answering" usually think in English, even when the question is in Portuguese. This project asks whether we can change that **without training the whole model**, using parameter-efficient fine-tuning (PEFT) techniques.

We start from [Lumi-4B](https://huggingface.co/ceia-br/lumi-4b), a 4-billion-parameter Brazilian model built by CEIA/UFG. The model it came from, NVIDIA Nemotron 3 Nano 4B, has a reasoning ("thinking") mode. In Lumi that mode was never trained or validated, and when it is turned on the model reasons in English. Our goal is to train a small adapter (LoRA or DoRA) so that, with thinking turned on, Lumi **reasons in Portuguese and answers more questions correctly**, without getting worse in its normal mode.

So far, with about 18 GPU-hours on a single RTX 4090:
- we got the model to reason in Portuguese in 93% to 100% of cases;
- it improved slightly in math;
- it loses 5 to 12 points on multiple-choice questions.

A reinforcement learning stage (GRPO and variants) recovered much of that loss, but we have not yet reached the success criterion we set before starting. This document presents the motivation, the experimental design, what we learned and what is left to test.

---

## 1. Motivation

### Why Portuguese?

Almost all language model research is done in English and for English. The small open models that Brazilian students, universities and companies can run on their own machines are often the ones that struggle most outside English. We want to understand, in practice, what happens when we try to bring an advanced capability (step-by-step reasoning) to **our native language**.

Reasoning written in Portuguese also has practical value: a student, teacher or professional can **read and check** the path the model took to its answer.

### Why PEFT?

Training all 4 billion parameters would take tens of gigabytes of GPU memory, far more than we have. PEFT techniques freeze the original model and train only a small set of new parameters. In our case that is about 17 million, less than 0.5% of the total. One training run takes 20 to 30 minutes on a consumer graphics card. This is what makes the study possible for an undergraduate group.

### Why this project?

Above all, this is a project for **learning**. We started with almost no experience in the field and wanted to follow the path of real research:
- ask a clear question;
- write hypotheses before running anything;
- fix the success criterion before seeing results;
- record everything, including what went wrong.

That is why every experiment in this repository has a hypothesis, a design, a result, a cost and a verdict.

---

## 2. Research question

> **Can a low-cost adapter make Lumi-4B reason in Portuguese, answer more correctly with reasoning turned on, and not get worse with reasoning turned off?**

Smaller questions follow from this one, each treated as its own study:

1. Is a prompt instruction enough, with no training at all?
2. Which training data works? Reasoning traces from which "teacher"?
3. Do hyperparameters matter? Learning rate, token limit, LoRA or DoRA?
4. **Where in the architecture** does the adapter help, and where does it hurt?
5. Can reinforcement learning with verifiable rewards fix what supervised fine-tuning broke?

We write each hypothesis down before its test and add the verdict afterwards. There are 26 so far (H0 to H25), in [estudos/07_hipoteses_e_proximos_passos.md](estudos/07_hipoteses_e_proximos_passos.md).

---

## 3. Key concepts

| Term | In plain words |
| --- | --- |
| **Fine-tuning** | Keep training a ready model on a small amount of data so it gets good at a specific task or behavior. |
| **SFT** (Supervised Fine-Tuning) | Fine-tuning with "question → ideal answer" examples. The model learns to imitate the answers. |
| **LoRA** | Instead of changing a large weight matrix **W**, learn a small correction **B·A** made of two thin matrices of rank *r* (here, *r* = 16). The original model stays frozen. |
| **DoRA** | A LoRA variant that splits each weight into a "size" (magnitude) and a "direction" and adjusts both. It promises to be closer to full fine-tuning. |
| **QLoRA** | LoRA with the base model stored in 4 bits, so it fits on smaller GPUs. It was in the initial plan but was taken out of scope: the BF16 model fits on our GPU, and quantization does not promise better quality. |
| **Thinking mode** | The model writes its reasoning between `<think>` and `</think>` before the final answer. |
| **Reasoning trace** | The text inside `<think>`. |
| **Teacher** | A larger model that writes reasoning traces for our model (the "student") to imitate. |
| **RL with verifiable rewards** | The model writes several answers, gets a "point" when it matches an answer key that can be checked automatically, and is updated to repeat what worked. We use GRPO and its variants GSPO, Dr. GRPO and RAFT. |

---

## 4. The model: Lumi-4B

| Item | Value |
| --- | --- |
| Origin | CEIA/UFG, built from NVIDIA Nemotron 3 Nano 4B |
| Original training | 50 billion tokens of continued pre-training (with distillation) + SFT in Portuguese, no RL |
| Version used | v0.2, commit `27b30f40` |
| Size | ~4 billion parameters, BF16, ~8 GB |
| Reasoning mode | Present in the chat template, but the model card asks to keep it off; when turned on, the model reasons in English |

### A different architecture

Lumi is **not a regular Transformer**. It is a **hybrid**. Of its 42 layers:
- 21 are **Mamba-2** blocks, a state-space model that reads text sequentially, like a modern RNN;
- 17 are **MLP** blocks;
- only 4 are **attention** blocks.

```
M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-
M = Mamba-2    - = MLP    * = attention
```

This matters for PEFT. Most LoRA tutorials put the adapter only on attention (`q_proj`, `v_proj`), which here would touch fewer than 10% of the layers. Our adapter covers **71 modules**:

| Block | Modules with LoRA | Left out (and why) |
| --- | --- | --- |
| Mamba-2 (21) | `in_proj` | `out_proj` and `conv1d`: the optimized kernel reads these weights directly and would ignore the adapter |
| MLP (17) | `up_proj`, `down_proj` | – |
| Attention (4) | `q_proj`, `k_proj`, `v_proj`, `o_proj` | – |

The hybrid architecture also raises a question that has not been explored much: **where should LoRA go in a model that mixes Mamba and attention?** We ran an ablation study on this (section 6).

---

## 5. Methodology

The full protocol is in [estudos/01_protocolo.md](estudos/01_protocolo.md). In short:

### 5.1 Training data

We tested several sources of Portuguese reasoning traces:

| Source | What it is | Use |
| --- | --- | --- |
| GigaVerbo-v2, `reasoning` subset | Short synthetic traces in Portuguese | First feasibility test |
| `adalberto-temp/gsm8k-sft-thinking-pt` | Qwen3-30B-A3B-Thinking traces for GSM8K problems, in Portuguese | Main SFT data (**no declared license: used for experiments only**) |
| Self-distillation | Lumi itself, forced to reason in Portuguese, filtered by the answer key | Test of a "weak teacher" |
| Jetson Thor packages | Traces from Nemotron 3 Nano 30B translated into Portuguese (and in English, for control), plus 12,000 translated questions for RL; a Qwen3-30B multiple-choice package is next | SFT runs `thor_pt`, `thor_en`, `thor_pt_short`; RL prompts |

Every example goes through filters:
- valid format, with exactly one `</think>`;
- an answer that matches the answer key or the majority of the teacher's answers;
- a trace in Portuguese;
- no overlap with the benchmarks.

For overlap we use two checks: an 8-gram index and a check for **duplicates in a different translation**, which compares the numbers in each problem and the words they share. The second check found that about a third of the original GSM8K validation set was repeated in the training data. Since then, validation uses only the 676 clean items.

### 5.2 Supervised fine-tuning (SFT)

- LoRA r = 16, alpha = 32, no dropout, learning rate 1e-4, 1 epoch, effective batch of 32 sequences.
- The loss covers only the answer tokens (trace + final answer).
- Our own training loop, without TRL, so we control exactly what goes into the loss.
- One run takes 20 to 30 min with LoRA and about 64 min with DoRA.

### 5.3 Reinforcement learning

Starting from the best SFT adapter, each cycle:
1. generates 8 answers for each of 256 questions (half math, half multiple choice);
2. gives a reward of 1 only when the answer closes `</think>`, is in Portuguese and matches the answer key;
3. updates the adapter.

One cycle takes 13 to 20 min. We compared GRPO, Dr. GRPO, GRPO with a KL penalty, RAFT (correct answers only) and GSPO (sequence-level ratio).

### 5.4 Evaluation

| Set | Size | Role |
| --- | --- | --- |
| GSM8K-PT, clean validation | 676 | Picking candidates (math) |
| Multiple-choice validation | 385 (Global-MMLU, MMLU-ProX, INCLUDE) | Picking candidates (knowledge and reasoning) |
| **Test suite** | GSM8K-PT, MMLU-ProX PT, ENEM, BLUEX, OAB | Final result only |

- Greedy decoding, up to 4,096 tokens, with vLLM.
- **Strict score:** the answer is read only after `</think>` and must be in the requested format (`#### N` or `Resposta: <letra>`).
- **Output shape:** whether the trace closes, whether it is in Portuguese, whether it gets truncated, and how many tokens it has.
- **Statistics:** paired bootstrap per item (10,000 resamples), with a 95% confidence interval for the difference against the original Lumi.

### 5.5 Success criterion (fixed before the experiments)

A candidate only "wins" if it does all of the following at once:
1. with reasoning on, scores higher than Lumi, with the confidence interval of the difference above zero;
2. closes `</think>` and answers in at least 98% of cases;
3. reasons in Portuguese in at least 90% of cases;
4. **loses no more than 1 point** with reasoning turned off.

One project rule is to **never change targets or thresholds to make a candidate pass**.

### 5.6 Infrastructure

- **Training and evaluation:** NVIDIA RTX 4090 (24 GiB), shared.
- **Data generation:** NVIDIA Jetson AGX Thor (122 GiB), used only to generate traces with larger teachers.
- **Code quality:** 286 automated CPU tests, covering answer extraction, data building, the training loop, adapter merging and GRPO.

---

## 6. Preliminary results

All numbers come from the **clean validation sets**, with strict scoring. ▲ and ▼ mean the 95% confidence interval of the difference against Lumi was above or below zero; no mark means a statistical tie. With 385 items, multiple-choice differences of up to 3 or 4 points are noise.

| Model | GSM8K, reasoning on | GSM8K, off | Multiple choice, reasoning on | Multiple choice, off | Trace in Portuguese |
| --- | --- | --- | --- | --- | --- |
| Original Lumi | 81.7 | 77.2 | 69.6 | 63.9 | 0% |
| Nemotron 3 Nano 4B (Lumi's parent) | 85.2 ▲ | 63.5 ▼ | 68.3 | 58.7 ▼ | ~0% |
| SFT on GigaVerbo (`viab`) | 67.2 ▼ | 57.7 ▼ | 43.4 ▼ | 43.1 ▼ | 100% |
| SFT on Qwen3-30B traces (`adalberto`) | 83.3 | 79.3 | 62.3 ▼ | 58.2 ▼ | 99.4% |
| SFT on self-distillation (`selfdist`) | 78.1 ▼ | 77.5 | 57.9 ▼ | 63.4 | 93.8% |
| SFT on Qwen3-30B traces, **DoRA** | 84.0 | 79.1 | 62.6 ▼ | 57.7 ▼ | 99.4% |
| SFT + RL with GSPO, 8 cycles (`gspo1`) | 84.3 ▲ | 78.6 | 67.5 | 57.7 ▼ | 96.1% |

Full table, generated automatically: [estudos/resultados_auto.md](estudos/resultados_auto.md).

### Main findings

1. **Asking is not enough.** Telling Lumi to "think in Portuguese" does not change the language of its reasoning. Forcing the start of the trace into Portuguese changes the language, but breaks the answer format.
2. **Portuguese is easy to learn; multiple choice gets lost.** Any SFT with good traces moves the reasoning into Portuguese and keeps or improves math. But multiple choice with reasoning drops 5 to 12 points.
3. **The teacher matters.** Traces from a stronger model (Qwen3-30B) work much better than traces from Lumi itself.
4. **LoRA vs. DoRA: a tie.** DoRA stayed within 1 point of LoRA on every measure and took 2.5 times longer to train. A lower learning rate, a higher token limit and scaling down the adapter did not fix the loss either.
5. **The language of the question is not the problem.** The original Lumi scores the same on questions in Portuguese and in English (1,200 questions, no significant difference).
6. **The adapter's location does not explain the loss.** Switching on only parts of an already trained adapter suggested that Mamba and attention carried the damage (the MLPs alone kept multiple choice at 69.6%). But **training** an adapter on the MLPs alone from scratch learned Portuguese (98.5%), improved math (83.4%) and lost the same on multiple choice (62.3%). The post-training screening was misleading; the likely cause is the training data.
7. **RL helps, but at a cost.** GRPO and its variants raised multiple choice with reasoning from 62.3% to 66–68%, a statistical tie with Lumi.
   - Only GSPO stayed stable; the other variants had entropy blow-ups or lost the output format.
   - Because RL trains only the reasoning mode, **the no-reasoning mode got worse**, dropping to 43.1% in two runs.

**So far, no candidate meets every success criterion.** The closest is `gspo1`. It meets the math gain and the language target, but not the `</think>` closing rate (93.2% vs. the required 98%), and it does not preserve the no-reasoning mode.

---

## 7. Discussion

Parameter-efficient fine-tuning turned out to be **cheap and powerful for changing the form** of the behavior: language, format and closing the reasoning. Changing the form **without damaging the content** is harder.

What we learned about the cause of the multiple-choice loss:
- **Hyperparameters:** they do not seem to be the cause. Learning rate, token limit, DoRA and adapter scale were tested and ruled out.
- **Language of the question:** not the cause either; the Portuguese vs. English diagnostic was a tie.
- **Teacher and data:** these seem to weigh more. The teacher matters, and the architecture does not explain it (an adapter on the MLPs alone loses the same). Where the answers are correct and the trace closes, math-only traces cost about 5 points of reasoning quality on multiple choice, while multiple-choice science traces do not (evidence, not proof).

Our main open hypothesis is the **teacher's style**. Lumi thinks in a short, plan-like way (median of 96 words), while Qwen3-30B writes longer, more "conversational" reasoning (177 words). Training Lumi to imitate a different way of thinking may cost it capability. We will test this with traces from a teacher in the **same family** (Nemotron 30B), translated into Portuguese.

It is also worth saying that well-measured negative results are results. Knowing that DoRA did not help here, that self-distillation makes the model worse, and that a contaminated validation set inflated scores by about 4 points is already useful for anyone working with models in Portuguese.

---

## 8. Limitations

- **One run per configuration.** We have not measured variation across seeds yet, so 1–2 point differences may be luck.
- **Small multiple-choice validation set** (385 items), with a confidence interval of ±4–5 points.
- **No candidate has been run on the final test suite.** Every number above comes from validation.
- **Trace language is measured by a heuristic** (counting Portuguese function words), not by a language detector. Human review covered only a few samples.
- **Answer keys in multiple-choice RL:** the first RL runs used the majority letter from another model's answers (DeepSeek-R1); the later ones use the dataset's own key (MedMCQA comes from exams; WebInstruct-verified labels were verified by a model, not by an exam board).
- **Several changes per RL run**, so the cause of each difference is not isolated.
- **The layer screening was done after training**, and it misled us: zeroing out parts of an adapter trained as a whole is not the same as training only those parts (the MLP-only retraining refuted it).
- **Licenses:** the `adalberto` dataset declares no license, and neither does Lumi's model card. Nothing trained on that data will be published.

---

## 9. Next steps

1. Separate the effect of the **generation limit** (2,048 vs. 4,096 tokens) from the effect of the **question set** in RL: a run with the official-key questions at 2,048 tokens is finishing now.
2. Train a **mixed SFT** with multiple-choice traces in Portuguese from a teacher that closes its reasoning cleanly (Qwen3-30B-A3B-Thinking, generated on the Jetson Thor), together with the Nemotron 30B data we already have, then screen it with a short RL run.
3. Run the best candidate **with reasoning on** on the test suite (GSM8K-PT, MMLU-ProX PT, ENEM, BLUEX, OAB), as a check only, and finish the original Lumi's test suite for comparison.
4. Repeat the best experiments with **other seeds**.
5. Adjust the GSPO recipe (sequences per minibatch, sampling temperature in the trainer, uniform sequence mean).
6. The adapter will only be used with reasoning on; with reasoning off the original Lumi runs. We will document that the merged model alone degrades the no-reasoning mode.

---

## 10. Repository layout

The study documents in `estudos/` and the PRD are written in Portuguese.

```text
PRD.md                 # requirements document: goal, targets, win criterion, phases
discovery.md           # engineering log and day-to-day decisions (index at the top)
comparacao.md          # comparison with reference models (local file, not versioned)
estudos/               # the project organized as ablation studies (start with its README)
  01_protocolo.md      # models, machines, training, evaluation, statistics
  02_sft_dados_e_professor.md     # SFT: data and teacher
  03_ablacao_camadas_e_mlp.md     # layer and MLP ablation
  04_rl_grpo.md                   # RL: GRPO and variants
  05_diagnosticos_e_referencias.md  # diagnostics and references
  06_registro_experimentos.md     # every run: configuration, time, folder
  07_hipoteses_e_proximos_passos.md # hypotheses and next steps
  08_registro_de_decisoes.md      # decisions of the day, with what to study if they go wrong
  resultados_auto.md   # table generated from the evaluation files
src/lumi_ft/           # code: data, modeling/PEFT, training, evaluation
scripts/               # training, RL, vLLM evaluation, merging, data generation (see scripts/README.md)
handoff/               # data-generation packages for the Jetson Thor (see handoff/README.md)
tests/                 # automated tests (CPU)
reports/               # data manifests and intermediate reports
env.sh, pyproject.toml, uv.lock   # environment (caches stay inside the project)
```

Not versioned: `data/`, `runs/` (adapters, evaluations, logs), `.venv*`, `.micromamba`, `tools/`, `comparacao.md`.

### Reproducing the numbers

The environment runs on Linux x86_64 with an NVIDIA GPU. `mamba-ssm` and `causal-conv1d` are compiled by `scripts/build_kernels.sh`; details are in [PRD.md](PRD.md).

```bash
source env.sh
PYTHONPATH=src python scripts/gen_estudos_resultados.py      # regenerates estudos/resultados_auto.md
python scripts/report_compare.py --ref Lumi \
    --col "Lumi=base:thinking" --col "gspo1=gspo1:thinking" \
    --tasks gsm8k_val,mcq_val --only-ids data/val_gsm8k_clean_ids.json   # comparison with 95% CI
PYTHONPATH=src python -m pytest tests --ignore tests/test_phase0_model.py  # CPU tests
```

Raw data, adapters and evaluations live in `data/` and `runs/`, outside Git.

---

## References

- Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- Liu, S.-Y. et al. (2024). *DoRA: Weight-Decomposed Low-Rank Adaptation*. [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)
- Dao, T. & Gu, A. (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2). [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
- Shao, Z. et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO). [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- Liu, Z. et al. (2025). *Understanding R1-Zero-Like Training: A Critical Perspective* (Dr. GRPO). [arXiv:2503.20783](https://arxiv.org/abs/2503.20783)
- Zheng, C. et al. (2025). *Group Sequence Policy Optimization* (GSPO). [arXiv:2507.18071](https://arxiv.org/abs/2507.18071)
- Dong, H. et al. (2023). *RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment*. [arXiv:2304.06767](https://arxiv.org/abs/2304.06767)
- *Tucano 2 Cool: Better Open Source LLMs for Portuguese* (2026). [arXiv:2603.03543](https://arxiv.org/abs/2603.03543)
- CEIA/UFG. *Lumi-4B*. [huggingface.co/ceia-br/lumi-4b](https://huggingface.co/ceia-br/lumi-4b)
