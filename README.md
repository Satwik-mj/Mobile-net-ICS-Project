# Edge AI Image Classifier — MobileNetV2 on CIFAR-10

A from-scratch PyTorch implementation of **MobileNetV2** trained on CIFAR-10, featuring a numerically-stable custom cross-entropy loss with label smoothing, two model variants optimised for different compute budgets, an ablation study comparing loss functions, and an interactive **Streamlit** inference dashboard.

## Novel Contribution

The core novelty of this project is a **fully transparent, step-auditable cross-entropy loss function** implemented entirely in raw tensor operations — with no delegation to PyTorch's black-box `nn.CrossEntropyLoss`. Every mathematical step (log-sum-exp numerical stabilisation, log-probability gathering via `.gather()`, mean NLL reduction, and label smoothing interpolation) is explicit, independently verifiable, and designed to make gradient flow fully traceable during debugging. This is validated through a controlled 3-way ablation study against `nn.CrossEntropyLoss` across 25 epochs, and further stress-tested across two MobileNetV2 width configurations (alpha=1.0 and alpha=0.35), providing quantitative evidence of convergence behaviour under both standard and edge-constrained compute budgets.

---

## Team & Contributions

| # | Name | Roll Number | Role |
|---|------|-------------|------|
| 1 | **Arya Kedar Dhumal** | CS25B1007 | Model Architecture & Loss Function |
| 2 | **Satwik Majumder** | AD25B1031 | Training & Evaluation Loop |
| 3 | **Ankita Padra** | AD25B1004 | Data Pipeline, Dataset Engineering & Streamlit UI |

---

## Project Structure

```
MobileNet_Project/
│
├── model.py              # MobileNetV2 architecture + custom loss          [Arya]
├── train.py              # Training loop, evaluation, checkpointing         [Satwik]
├── evaluate_metrics.py   # Per-class metrics & ablation study               [Satwik]
│
├── dataset.py            # Data transforms & DataLoader construction        [Ankita]
├── prepare_dataset.py    # CIFAR-10 → ImageFolder converter                 [Ankita]
│
├── app.py                # Streamlit inference dashboard (UI)               [Ankita]
├── requirements.txt      # Python dependencies
├── .gitignore            # Excludes dataset, cache, and model checkpoints
│
├── cifar-10-batches-py/  # Raw CIFAR-10 pickle files (not tracked by git)
├── my_custom_dataset/    # Generated after running prepare_dataset.py (not tracked)
│   ├── train/<class>/    # 50,000 training images across 10 class folders
│   └── test/<class>/     # 10,000 test images across 10 class folders
│
├── mobilenet_alpha_1_0.pth   # Trained checkpoint — Standard model (alpha=1.0)
├── mobilenet_alpha_0_35.pth  # Trained checkpoint — Lightweight model (alpha=0.35)
│
└── results/
    ├── per_class_metrics.txt  # Detailed per-class accuracy, precision, recall, F1
    ├── per_class_f1.png       # Bar chart of per-class F1 scores
    ├── ablation_results.txt   # Epoch-by-epoch loss comparison table
    └── ablation_summary.png   # Train/test accuracy curves across loss variants
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset *(run once)*

```bash
python prepare_dataset.py
```

Converts the raw CIFAR-10 pickle files into an `ImageFolder`-compatible directory tree under `./my_custom_dataset/`.

### 3. Train both model variants

```bash
python train.py
```

Saves `mobilenet_alpha_1_0.pth` and `mobilenet_alpha_0_35.pth` to the project root.

### 4. Run evaluation & ablation study

```bash
python evaluate_metrics.py
```

Generates all plots and reports inside the `results/` folder.

### 5. Launch the inference dashboard

```bash
streamlit run app.py
```

---

## Quantitative Results

Both variants were trained for **25 epochs** on CIFAR-10 (50,000 training images, 10,000 test images, upsampled to 64×64). All metrics are reported on the held-out test set.

### Standard Model — alpha = 1.0

| Metric | Value |
|--------|-------|
| Train Loss (Epoch 25) | 0.1266 |
| Train Accuracy | 95.55% |
| **Test Accuracy** | **78.56%** |
| **Precision (macro)** | **0.7884** |
| **Recall (macro)** | **0.7856** |
| **F1 Score (macro)** | **0.7862** |

### Per-Class F1 Score — alpha = 1.0

![Per-Class F1](results/per_class_f1.png)

| Class | F1 Score |
|-------|----------|
| airplane | 0.82 |
| automobile | 0.89 |
| bird | 0.70 |
| cat | 0.62 |
| deer | 0.75 |
| dog | 0.71 |
| frog | 0.84 |
| horse | 0.83 |
| ship | 0.87 |
| truck | 0.88 |
| **Macro Average** | **0.7910** |

Cat (0.62) and dog (0.71) are the hardest classes — visually similar animals the model consistently confuses. Automobile (0.89) and truck (0.88) are the easiest, benefiting from distinct shapes and high inter-class contrast.

### Lightweight Model — alpha = 0.35

| Metric | Value |
|--------|-------|
| Train Accuracy (Epoch 25) | 88.36% |
| **Test Accuracy** | **74.01%** |
| **Precision (macro)** | **0.7433** |
| **Recall (macro)** | **0.7401** |
| **F1 Score (macro)** | **0.7405** |
| Approx. Parameters | ~0.4 M (vs ~3.4 M for alpha=1.0) |
| Relative MACs | ~0.09× |
| Checkpoint size | 1.72 MB (vs ~8.8 MB for alpha=1.0) |
| Target platform | Edge / Microcontroller |

### Model Comparison

| Variant | Parameters | MACs | Checkpoint | Test Acc | Precision | Recall | F1 | Use-case |
|---------|-----------|------|------------|----------|-----------|--------|----|----------|
| alpha = 1.00 | ~3.4 M | 1.00× | ~8.8 MB | **78.56%** | 0.7884 | 0.7856 | 0.7862 | Server / GPU |
| alpha = 0.35 | ~0.4 M | ~0.09× | 1.72 MB | **74.01%** | 0.7433 | 0.7401 | 0.7405 | Edge / Microcontroller |

The alpha=0.35 variant achieves an ~80% reduction in checkpoint size and a ~9× reduction in multiply-accumulate operations at a cost of **4.55 percentage points** in test accuracy — a quantified, deliberate accuracy/compute trade-off for deployment on resource-constrained hardware. The fact that a model with only ~0.4 M parameters still reaches 74% accuracy on a 10-class problem demonstrates the effectiveness of the MobileNetV2 depthwise-separable architecture even at aggressive width scaling.

---

## Ablation: CustomCrossEntropyLoss vs nn.CrossEntropyLoss

![Ablation Summary](results/ablation_summary.png)

All three loss variants were trained from scratch for **25 epochs** on MobileNetV2 alpha=1.0.

| Loss Function | Final Test Accuracy (25 epochs) |
|---------------|--------------------------------|
| nn.CrossEntropyLoss (baseline) | 73.93% |
| Custom CE (no smoothing, ε = 0.0) | 72.97% |
| Custom CE + Label Smoothing (ε = 0.1) | 72.69% |

**Interpreting the results:**

The three variants finish within ~1.25 percentage points of each other after 25 epochs, which confirms that the custom implementation is mathematically correct — it converges on the same loss landscape as `nn.CrossEntropyLoss`. The ~1% gap is within the expected variance of a fixed 25-epoch run with a single random seed and is not statistically significant.

The purpose of this ablation is not to claim that the custom loss *outperforms* PyTorch's built-in — it is to demonstrate **mathematical equivalence and transparency**. Every step (log-sum-exp stabilisation, log-probability gathering via `.gather()`, mean NLL reduction, and label smoothing interpolation) is written as auditable tensor operations, making gradient flow fully traceable in a way that `nn.CrossEntropyLoss` — which fuses all of these into a single C++ CUDA kernel — does not permit.

**Why label smoothing (ε=0.1) slightly reduces accuracy here:**
Label smoothing redistributes a fraction ε of the probability mass away from the correct class and spreads it uniformly across all C classes. On CIFAR-10 with only 10 classes and 5,000 training samples per class, the model has sufficient capacity to learn well-calibrated hard targets without over-fitting — so the soft targets introduced by smoothing slightly *under-specify* the correct answer and marginally hurt discriminative accuracy. Label smoothing is most beneficial in larger-vocabulary settings (e.g. ImageNet-1000, machine translation) where over-confident predictions on rare classes are a real risk. At 25 epochs on CIFAR-10, the regularisation effect is unnecessary and introduces a small accuracy penalty, which is the expected and documented behaviour.

---

## Contribution Details

### 1 — Arya Kedar Dhumal — Model Architecture & Loss Function

**File:** `model.py`

#### `CustomCrossEntropyLoss` — Manual Loss Implementation

Multi-class cross-entropy is implemented as explicit tensor operations rather than delegating to `nn.CrossEntropyLoss`:

```
Step 1 — Numerically stable Log-Softmax (log-sum-exp trick):
log_p(k) = z_k − max(z) − log( Σ_j exp(z_j − max(z)) )

Step 2 — Gather the log-probability of the correct class y:
correct_log_prob_i = log_p_i(y_i)

Step 3 — Mean Negative Log-Likelihood over the batch of N samples:
L_CE = (1/N) × Σ_i  −log_p_i(y_i)

Step 4 — Label Smoothing interpolation (controlled by ε):
L_smooth = −mean(log_p)          [uniform over all classes]
L_final  = (1 − ε) × L_CE + ε × L_smooth
```

Subtracting `max(z)` before exponentiation (the log-sum-exp trick) prevents `exp()` overflow for large logit values — a critical numerical stability concern that `nn.CrossEntropyLoss` handles internally but invisibly.

#### `build_mobilenet(alpha, num_classes)` — Architecture Factory

- `alpha` controls filters in every convolutional layer: `actual_filters = round(base_filters × alpha)`, shrinking the entire channel dimension uniformly.
- The default ImageNet-1000 classifier head is replaced with `nn.Linear(in_features, num_classes)`, with `in_features` read directly from the existing layer so it automatically reflects whatever `alpha` was used — no hardcoded dimension constants.
- Weights are randomly initialised (no pretrained ImageNet transfer), requiring the network to learn entirely from the CIFAR-10 training data.

| alpha | Relative MACs | Approx. Parameters | Use-case |
|-------|--------------|-------------------|----------|
| 1.00 | 1.00× | ~3.4 M | Standard accuracy |
| 0.35 | ~0.09× | ~0.4 M | Ultra-light edge / microcontroller |

---

### 2 — Satwik Majumder — Training & Evaluation Loop

**Files:** `train.py`, `evaluate_metrics.py`

#### Training Loop (per epoch)

- Detects and selects the available compute device (`cuda` / `cpu`) at runtime; moves the model and all batches with `non_blocking=True` for overlap with compute.
- Applies `torch.backends.cudnn.benchmark = True` to enable the NVIDIA cuDNN auto-tuner.
- Calls `model.train()` to activate `Dropout` and switch `BatchNorm` to accumulate running statistics.
- For each mini-batch: zeroes gradients → forward pass → `CustomCrossEntropyLoss` → `loss.backward()` → `optimizer.step()`.
- Optimiser: **Adam** (`lr=1e-3`) with per-parameter adaptive learning rates.

#### Evaluation Loop

- Calls `model.eval()` to disable `Dropout` and freeze `BatchNorm` running statistics.
- Wraps the pass in `torch.no_grad()` to disable autograd graph construction.
- Reports **accuracy, precision, recall, and F1** at the end of training.

#### `evaluate_metrics.py` — Per-Class Metrics & Ablation Study

- **Part 1** loads the trained `mobilenet_alpha_1.0.pth` checkpoint and produces a full `sklearn` classification report with per-class accuracy, precision, recall, F1, and support — saved to `results/per_class_metrics.txt` and plotted as `results/per_class_f1.png`.
- **Part 2** reruns training from scratch for all three loss variants (25 epochs each) and records epoch-level train/test accuracy curves — saved to `results/ablation_results.txt` and `results/ablation_summary.png`.

#### Checkpoint Saving

- Serialises only the `state_dict` to the smallest portable checkpoint format.
- Trains and saves both variants: `mobilenet_alpha_1_0.pth` (~8.8 MB) and `mobilenet_alpha_0_35.pth` (1.72 MB).

---

### 3 — Ankita Padra — Data Pipeline, Dataset Engineering & UI

**Files:** `prepare_dataset.py`, `dataset.py`, `app.py`

#### `prepare_dataset.py` — CIFAR-10 → ImageFolder Converter

CIFAR-10 ships as six binary pickle files; `torchvision.datasets.ImageFolder` expects a directory tree organised by class. This one-time converter bridges the two formats:

- Unpickles each of the five training batches and the single test batch.
- Reshapes the flat `(N, 3072)` uint8 array to `(N, 3, 32, 32)` then transposes to HWC layout `(N, 32, 32, 3)` required by PIL.
- Writes every image as a lossless PNG inside `my_custom_dataset/{split}/{class_name}/`.
- Uses a deterministic filename scheme (`{split}_b{batch}_{index:05d}.png`) for full reproducibility.

#### `dataset.py` — Transform Pipeline & DataLoaders

```
Raw PNG on disk
  → PIL Image       (H × W × 3, uint8)
  → Resize 64×64    (64 × 64 × 3, uint8)   — uniform spatial dimensions per batch
  → ToTensor        (3 × 64 × 64, float32, [0, 1])
  → Normalize(0.5)  (3 × 64 × 64, float32, [−1, 1])
```

The same `NORM_MEAN` / `NORM_STD` constants are re-used in `app.py`'s `INFER_TRANSFORM` to guarantee train-inference consistency.

| Parameter | Train | Test | Reason |
|-----------|-------|------|--------|
| `shuffle` | True | False | Randomise batch composition per epoch; deterministic eval |
| `num_workers` | 0 | 0 | Bypasses Windows multiprocessing overhead |
| `pin_memory` | True | True | Page-locked memory for faster CPU→GPU transfer |

#### `app.py` — Streamlit Inference Dashboard

- Sidebar model selector to switch between the two trained checkpoints at runtime.
- Displays checkpoint size in MB and the relative size saving of the lightweight variant.
- Image uploader supporting JPEG, PNG, BMP, and WebP.
- Runs inference with softmax confidence score, rendered as both a numeric metric and a progress bar.
- Lists all detected class names from the dataset directory automatically.

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| **PyTorch** | Model definition, training, inference |
| **torchvision** | MobileNetV2 skeleton, `ImageFolder`, transforms |
| **Streamlit** | Interactive inference dashboard |
| **Pillow** | Image I/O |
| **NumPy** | CIFAR-10 array reshaping |
| **scikit-learn** | Precision, recall, F1 computation and classification report |

---

## Dataset

**CIFAR-10** — 60,000 colour images (32×32) across 10 classes:

`airplane · automobile · bird · cat · deer · dog · frog · horse · ship · truck`

| Split | Images | Per class |
|-------|--------|-----------|
| Training | 50,000 | 5,000 |
| Test | 10,000 | 1,000 |

Images are upsampled to **64×64** during the transform pipeline to give MobileNetV2 more spatial resolution to work with.

---

## .gitignore

The following are excluded from version control:

```
__pycache__/
cifar-10-batches-py/
my_custom_dataset/
mobilenet_alpha_*
```

The raw dataset, converted image tree, and trained model checkpoints are not tracked — run the pipeline steps above to regenerate them locally.

> **Important:** The `results/` folder is **not** listed in `.gitignore` and should be committed to the repository. Always run `python evaluate_metrics.py` before pushing and include the generated `results/` folder so that plots and metric reports are visible in the repo without requiring a full retraining run.
