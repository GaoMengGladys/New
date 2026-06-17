# Skim and Skip (SAS)

<p align="center">
  <a href="https://github.com/GaoMengGladys/SAS"><img src="https://img.shields.io/badge/Project-Page-green?style=flat-square" alt="Project Page"></a>
  <a href="https://huggingface.co/MgGladys"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Profile-blue?style=flat-square" alt="Hugging Face Profile"></a>
  <a href="https://huggingface.co/datasets/TIGER-Lab/MMEB"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-green?style=flat-square" alt="Hugging Face Data"></a>
  <a href="https://github.com/GaoMengGladys/SAS/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-orange?style=flat-square" alt="License"></a>
</p>

**Skim and Skip: Efficient Multimodal Retrieval via Hierarchical Adaptive Inference**

SAS is a hierarchical adaptive inference framework for efficient multimodal retrieval. It reduces the computational cost of MLLM-based retrievers by making two sequential decisions:

1. **Skim** (Token-level Evidence Selection): Identifies and preserves the input tokens that genuinely contribute to the final [EOS] retrieval embedding using a theoretically-derived influence score, removing redundant tokens at an intermediate layer.

2. **Skip** (Layer-level Depth Allocation): Evaluates whether the current intermediate representation is already sufficient for reliable retrieval via a lightweight sufficiency classifier, routing easy queries to early exit while allowing hard queries to continue through deeper layers.

SAS retains ~99% of the dense baseline's retrieval performance while achieving **1.64x** end-to-end speedup and ~**3x** FLOPs reduction on 12 MMEB retrieval benchmarks.

---

## 🤗 Model Zoo

We release our trained SAS backbone models and early-exit sufficiency classifiers on Hugging Face:

| Model Name | Type | Hugging Face Download |
| :--- | :---: | :---: |
| **SAS 3B** (Backbone) | Backbone Model | [<img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SAS__3B-blue?style=flat-square">](https://huggingface.co/MgGladys/SAS_3B) |
| **SAS 3B Classifier** | Sufficiency Classifier | [<img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Classifier-orange?style=flat-square">](https://huggingface.co/MgGladys/SAS_3B_Classifier) |
| **SAS 7B** (Backbone) | Backbone Model | [<img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SAS__7B-blue?style=flat-square">](https://huggingface.co/MgGladys/SAS_7B) |
| **SAS 7B Classifier** | Sufficiency Classifier | [<img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Classifier-orange?style=flat-square">](https://huggingface.co/MgGladys/SAS_7B_Classifier) |

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies:
- `transformers>=4.52.0`
- `peft`
- `accelerate`
- `torch` (with CUDA and flash-attention-2 support)
- `qwen-vl-utils[decord]==0.0.8`

For flash-attention:
```bash
pip install flash-attn --no-build-isolation
```

## Data Preparation

### Training Data

We use the retrieval subset of [MMEB](https://huggingface.co/datasets/TIGER-Lab/MMEB) for training, which includes 8 datasets (~620K samples total): VisDial, CIRR, VisualNews (i2t/t2i), MSCOCO (i2t/t2i), NIGHTS, and WebQA.

The dataset configuration is in `configs/train_image_ret.yaml`. Each dataset is loaded from HuggingFace Hub automatically.

### Evaluation Data

We evaluate on 12 image retrieval tasks from MMEB. Download the evaluation data:

```bash
mkdir -p data/MMEB-V2
# Follow MMEB-V2 instructions to download evaluation datasets
# See: https://huggingface.co/datasets/TIGER-Lab/MMEB
```

The evaluation configuration is in `configs/eval_image_retrieval.yaml`. Update `data_basedir` in the eval script to point to your data directory.

## Training

SAS uses a two-stage training pipeline:

### Stage I: Multi-layer Supervised Training with Knowledge Distillation

Train the backbone model with multi-layer contrastive supervision and knowledge distillation to activate retrieval capability at the intermediate layer.

```bash
bash scripts/train_sas_stage1.sh
```

Key configurations:
| Parameter | Description | Default |
|---|---|---|
| `--supervise_layers` | Layers for multi-layer supervision | `"12,-1"` (layer 12 + last) |
| `--supervise_weights` | Weight for each supervised layer | `"0.1,0.9"` |
| `SKIM_LAYER` | Layer for token selection | `12` |
| `SKIM_KEEP_RATIO_TEXT` | Fraction of text tokens to keep | `0.5` |
| `SKIM_VPOOL_KERNEL` | Vision pooling kernel size | `2` (4x reduction) |
| `KD_ENABLED` | Enable knowledge distillation | `1` |
| `KD_TYPE` | Distillation type | `cosine` |

The training schedule gradually shifts from pure distillation to contrastive learning via `CE_STAGE1`, `CE_STAGE2`, and `CE_FINAL`.

### Stage II: Sufficiency Classifier Training

Freeze the Stage I backbone and train a lightweight binary classifier to predict whether the mid-layer representation is sufficient for retrieval.

```bash
# Update STAGE1_CKPT in the script to point to your Stage I checkpoint
bash scripts/train_sas_stage2.sh
```

Key configurations:
| Parameter | Description | Default |
|---|---|---|
| `SKIP_LAYER` | Layer for early exit decision | `12` |
| `--checkpoint_path` | Path to Stage I checkpoint | Required |
| `--learning_rate` | Classifier learning rate | `5e-4` |
| `--max_steps` | Training steps | `1000` |
| `--per_device_train_batch_size` | Batch size | `512` |

## Evaluation

### SAS Inference (Skim + Skip)

Run the full SAS pipeline with token pruning and adaptive early exit:

```bash
# Update paths in the script
bash scripts/eval_sas.sh
```

The `SKIP_THRESHOLD` controls the exit aggressiveness (lower = more queries exit early = faster but potentially less accurate). Per-dataset thresholds are defined in `eval_sas.py`.

### Dense Baseline

Run the dense baseline (full model, no Skim/Skip) for comparison:

```bash
bash scripts/eval_dense.sh
```

## Project Structure

```
SAS/
├── train_stage1.py          # Stage I: Multi-layer supervised training
├── train_stage2.py          # Stage II: Sufficiency classifier training
├── eval_sas.py              # SAS evaluation (Skim + Skip)
├── eval_dense.py            # Dense baseline evaluation
├── requirements.txt
├── configs/
│   ├── train_image_ret.yaml       # Training dataset config
│   └── eval_image_retrieval.yaml  # Evaluation dataset config
├── scripts/
│   ├── train_sas_stage1.sh  # Stage I training script
│   ├── train_sas_stage2.sh  # Stage II training script
│   ├── eval_sas.sh          # SAS evaluation script
│   └── eval_dense.sh        # Dense baseline script
└── src/
    ├── arguments.py         # Training/model/data arguments
    ├── loss.py              # Multi-layer contrastive loss + distillation
    ├── trainer.py           # Stage I trainer (GradCache + KD schedule)
    ├── classifier_trainer.py # Stage II classifier trainer
    ├── classifier.py        # EarlyExitClassifier model
    ├── utils.py             # Utilities
    ├── dist_utils.py        # Distributed training utilities
    ├── model/
    │   ├── sas_model.py     # MMEBModel with Skim/Skip support
    │   ├── processor.py     # Qwen2.5-VL processor
    │   └── qwen2_5_vl/      # Modified Qwen2.5-VL backbone with:
    │       │                 #   - SKIMTokenPruner (influence-based scoring)
    │       │                 #   - Vision token pooling
    │       │                 #   - stop_at_layer / resume_state (early exit)
    │       └── modeling_qwen2_5_vl.py
    ├── data/                # Dataset loading and collation
    ├── eval_utils/          # Ranking metrics (Hit@K, NDCG, MRR, etc.)
    ├── grad_cache/          # Gradient caching for memory-efficient training
    └── prompt/              # Prompt templates
```

## Key Environment Variables

### Skim (Token Pruning)
| Variable | Description |
|---|---|
| `SKIM_ENABLED` | Enable token pruning (0/1) |
| `SKIM_LAYER` | Layer at which pruning occurs |
| `SKIM_SELECTION` | Selection method: `skim` (influence score), `attention`, `random` |
| `SKIM_KEEP_RATIO_TEXT` | Fraction of text tokens to keep |
| `SKIM_KEEP_RATIO_VISION` | Fraction of vision tokens to keep |
| `SKIM_APPLY` | Apply to `qry`, `tgt`, or `both` |

### Vision Pooling
| Variable | Description |
|---|---|
| `SKIM_VPOOL_ENABLED` | Enable vision token pooling (0/1) |
| `SKIM_VPOOL_LAYER` | Layer for pooling (default: 1) |
| `SKIM_VPOOL_KERNEL` | Pooling kernel size (default: 2) |

### Skip (Early Exit)
| Variable | Description |
|---|---|
| `SKIP_ENABLED` | Enable early exit (0/1) |
| `SKIP_LAYER` | Layer for exit decision |
| `SKIP_METHOD` | Exit method: `classifier` |
| `SKIP_THRESHOLD` | Confidence threshold for exit |
| `SKIP_CLASSIFIER_PATH` | Path to trained classifier |


## Acknowledgements

This codebase is built upon [VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec). The backbone model is [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL).


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
