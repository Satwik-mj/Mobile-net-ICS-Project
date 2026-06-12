# Edge AI Image Classifier -- MobileNetV2 on CIFAR-10

A from-scratch PyTorch implementation of **MobileNetV2** trained on CIFAR-10, featuring a custom cross-entropy loss, two model variants optimised for different compute budgets, and an interactive **Streamlit** inference dashboard.

---

## Team & Contributions

| # | Name | Roll Number | Role |
| --- | --- | --- | --- |
| 1 | **Arya Kedar Dhumal** | CS25B1007 | Model Architecture & Loss Function |
| 2 | **Satwik Majumder** | AD25B1031 | Training & Evaluation Loop |
| 3 | **Ankita Padra** | AD25B1004 | Data Pipeline & Dataset Engineering |

---

## Project Structure

```
MobileNet_Project/
|
|-- model.py              # MobileNetV2 architecture + custom loss  [Arya]
|-- train.py              # Training loop, evaluation, checkpointing [Satwik]
|
|-- dataset.py            # Data transforms & DataLoader construction [Ankita]
|-- prepare_dataset.py    # CIFAR-10 -> ImageFolder converter         [Ankita]
|
|-- app.py                # Streamlit inference dashboard (UI)        [Ankita]
|-- requirements.txt      # Python dependencies
|
|-- cifar-10-batches-py/  # Raw CIFAR-10 pickle files
`-- my_custom_dataset/    # Generated after running prepare_dataset.py
    |-- train/<class>/    # 50 000 training images across 10 class folders
    `-- test/<class>/     # 10 000 test images across 10 class folders

```

---

## Contribution Details

### 1 - Arya Kedar Dhumal -- Model Architecture & Loss Function

**File:** `model.py`

Arya designed and implemented the two core components that define *what the network is* and *how it measures error*.

**`CustomCrossEntropyLoss` -- Manual Loss Implementation**

Rather than delegating to `nn.CrossEntropyLoss`, Arya implemented multi-class cross-entropy as explicit tensor operations so every mathematical step is transparent and auditable:

```
Step 1 -- Numerically stable Log-Softmax (log-sum-exp trick):
    log_p(k) = z_k - max(z) - log( Sum_j exp(z_j - max(z)) )

Step 2 -- Gather the log-probability of the correct class y:
    correct_log_prob_i = log_p_i(y_i)

Step 3 -- Mean Negative Log-Likelihood over the batch of N samples:
    L = (1/N) * Sum_i  -log_p_i(y_i)

```

Subtracting `max(z)` before exponentiation (the log-sum-exp trick) prevents `exp()` overflow for large logit values -- a critical numerical stability concern that `nn.CrossEntropyLoss` handles internally but invisibly.

**`build_mobilenet(alpha, num_classes)` -- Architecture Factory**

Arya implemented the MobileNetV2 factory function that instantiates the network scaled by the **width multiplier alpha**:

* `alpha` controls the number of filters in every convolutional layer: `actual_filters = round(base_filters * alpha)`, shrinking the entire channel dimension uniformly throughout the network.
* The default ImageNet-1000 classifier head is replaced with a custom `nn.Linear(in_features, num_classes)` layer, while the upstream `Dropout` is preserved for regularisation. `in_features` is read directly from the existing layer so it automatically reflects whatever `alpha` was used -- no hardcoded dimension constants.
* Weights are randomly initialised (no pretrained ImageNet transfer), requiring the network to learn entirely from the CIFAR-10 training data.

| alpha | Relative MACs | Approx. Parameters | Use-case |
| --- | --- | --- | --- |
| 1.00 | 1.00x | ~3.4 M | Standard accuracy |
| 0.35 | ~0.09x | ~0.4 M | Ultra-light edge / microcontroller |

---

### 2 - Satwik Majumder -- Training & Evaluation Loop

**File:** `train.py`

Satwik built the orchestration layer that defines *how the network learns* -- connecting the model and loss function from `model.py` to the data stream from `dataset.py`, and driving the full optimisation process.

**Device & Setup**

* Detects and selects the available compute device (`cuda` / `cpu`) at runtime, moves the model and all batches to that device using `non_blocking=True` transfers for overlap with compute.
* Instantiates the `Adam` optimizer (`lr=1e-3`), which maintains per-parameter adaptive learning rates using first and second moment estimates of the gradient:
```
theta = theta - lr * m_hat / (sqrt(v_hat) + epsilon)

```



**Training Loop (per epoch)**

* Calls `model.train()` to activate `Dropout` and switch `BatchNorm` to accumulate running statistics from the current batch.
* For each mini-batch: zeroes accumulated gradients (`optimizer.zero_grad()`), runs the forward pass to get logits `(B, C)`, computes the `CustomCrossEntropyLoss`, calls `loss.backward()` to compute `dL/d_theta` via autograd for every learnable parameter, then applies `optimizer.step()` to update weights.
* Accumulates running loss and correct predictions to report epoch-level train accuracy.

**Evaluation Loop (per epoch)**

* Calls `model.eval()` to disable `Dropout` and switch `BatchNorm` to use frozen running statistics.
* Wraps the evaluation pass in `torch.no_grad()` to disable autograd graph construction, reducing memory usage and speeding up inference.
* Reports test accuracy after every epoch so convergence can be monitored.

**Checkpoint Saving**

* Serialises only the `state_dict` (parameter and buffer tensors, no optimizer state) to the smallest portable checkpoint format.
* Trains and saves both variants back-to-back: `mobilenet_alpha_1.0.pth` and `mobilenet_alpha_0.35.pth`.

---

### 3 - Ankita Padra -- Data Pipeline & Dataset Engineering

**Files:** `prepare_dataset.py`, `dataset.py` *(+ `app.py` -- Streamlit UI, ungraded)*

Ankita owned the complete data layer -- everything from raw binary files on disk to batched, normalised tensors ready for the GPU.

**`prepare_dataset.py` -- CIFAR-10 -> ImageFolder Converter**

CIFAR-10 ships as six binary pickle files; `torchvision.datasets.ImageFolder` expects a directory tree of images organised by class. Ankita wrote the one-time converter that bridges these two formats:

* Unpickles each of the five training batches and the single test batch using `pickle.load(encoding="bytes")`.
* Reshapes CIFAR-10's flat `(N, 3072)` uint8 array to `(N, 3, 32, 32)` then transposes channel-first to HWC layout `(N, 32, 32, 3)` required by PIL.
* Writes every image as a lossless PNG inside `my_custom_dataset/{split}/{class_name}/`, creating directories automatically with `os.makedirs(exist_ok=True)`.
* Uses a deterministic filename scheme (`{split}_b{batch}_{index:05d}.png`) so images from different batches never overwrite each other and the dataset can be fully regenerated reproducibly.
* Prints a per-class image count summary on completion, making it easy to verify dataset integrity.

**`dataset.py` -- Transform Pipeline & DataLoaders**

Ankita defined the canonical transformation pipeline that every image passes through, and configured the `DataLoader` instances that batch and stream data to the training loop:

Transform pipeline (identical for train and test):

```
Raw PNG/JPEG on disk
    -> PIL Image      (H x W x 3,  uint8)
    -> Resize 64x64   (64 x 64 x 3, uint8)   -- uniform spatial dimensions per batch
    -> ToTensor       (3 x 64 x 64, float32, [0, 1])
    -> Normalize(0.5, 0.5)  (3 x 64 x 64, float32, [-1, 1])

```

Normalising to `[-1, 1]` centres the activation distribution around zero, which stabilises gradient magnitude during early training. The same `NORM_MEAN` / `NORM_STD` constants are re-used in `app.py`'s `INFER_TRANSFORM` to guarantee train-inference consistency.

DataLoader configuration:

* `shuffle=True` for training -- randomises batch composition each epoch, reducing the risk of the model memorising batch ordering.
* `shuffle=False` for evaluation -- deterministic order ensures reproducible accuracy metrics across runs.
* `num_workers=2` -- parallel disk I/O workers to overlap data loading with GPU compute.
* `pin_memory=True` -- allocates tensors in page-locked memory for faster CPU -> GPU transfer.

Exposes `class_names` (alphabetically sorted sub-directory names from `ImageFolder`) as the consistent label-to-index mapping used by both the training loop and the inference dashboard.

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

### 3. Train both model variants

```bash
python train.py

```

Saves `mobilenet_alpha_1.0.pth` and `mobilenet_alpha_0.35.pth`.

### 4. Launch the inference dashboard

```bash
streamlit run app.py

```

---

## Model Comparison

| Metric | Standard (alpha = 1.0) | Lightweight (alpha = 0.35) |
| --- | --- | --- |
| Width multiplier | 1.00 | 0.35 |
| Approx. parameters | ~3.4 M | ~0.4 M |
| Relative MACs | 1.00x | ~0.09x |
| Target platform | Server / GPU | Edge / Microcontroller |

---

## Tech Stack

* **PyTorch** -- model definition, training, inference
* **torchvision** -- MobileNetV2 skeleton, `ImageFolder`, transforms
* **Streamlit** -- interactive inference dashboard
* **Pillow** -- image I/O
* **NumPy** -- CIFAR-10 array reshaping

---

## Dataset

**CIFAR-10** -- 60 000 colour images (32 x 32) across 10 classes:

`airplane - automobile - bird - cat - deer - dog - frog - horse - ship - truck`

* Training split: 50 000 images (5 000 per class)
* Test split: 10 000 images (1 000 per class)

Images are upsampled to **64 x 64** during the transform pipeline to give MobileNetV2 more spatial resolution.
