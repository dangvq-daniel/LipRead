# LipRead Project – Enhanced Version

This project is an improved version of the original LipRead codebase, featuring **preprocessing optimizations**, **memory-efficient model architecture**, **advanced training callbacks**, and **GPU memory optimizations**.

---

## 1. Preprocessing Changes

**Original:**

* Data loaded on the fly from `.mpg` video files.
* `tf.data.Dataset` mapped directly from video paths and alignment files.

**Modified:**

* Preprocessed all videos into `.npz` batches (`frames` + `labels`) for faster loading.
* Preprocessing code:

```python
export_dir = './preprocessed_full'
os.makedirs(export_dir, exist_ok=True)

for i, (frames, labels) in enumerate(data):
    np.savez(os.path.join(export_dir, f'batch_{i}.npz'),
             frames=frames.numpy(), labels=labels.numpy())
```

* Data loading now reads `.npz` files, converts them to tensors, and creates a `tf.data.Dataset`:

```python
files = sorted(glob.glob('.././preprocessed_full/batch_*.npz'))
dataset_list = []
for f in files:
    batch = np.load(f)
    frames = tf.convert_to_tensor(batch['frames'], dtype=tf.float32)
    labels = tf.convert_to_tensor(batch['labels'], dtype=tf.int64)
    dataset_list.append((frames, labels))

frames_all = tf.concat([b[0] for b in dataset_list], axis=0)
labels_all = tf.concat([b[1] for b in dataset_list], axis=0)

data = tf.data.Dataset.from_tensor_slices((frames_all, labels_all))
data = data.batch(2).prefetch(tf.data.AUTOTUNE)
```

* This avoids repeated video decoding and accelerates training significantly.

---

## 2. Model Architecture Changes

**Original:**

* 3D-CNN layers with filters `[128, 256, 75]`.
* BiLSTM layers with 128 units.
* Dense output layer with softmax.

**Modified:**

* **Memory-efficient 3D-CNN + BiLSTM model**:

  * Reduced convolution filters to `[32, 64, 128]`.
  * BiLSTM layers replaced with `RNN(LSTMCell(...))` on CPU for reduced GPU memory usage.
  * Added **residual connections**, **batch normalization**, and **layer normalization**.
  * BiLSTM units set to 128; dropout 0.3.
* Output layer: `Dense(vocab_size, activation='softmax')`.
* Training on GPU now uses less memory, and **training time per epoch reduced to ~12 minutes**.

---

## 3. Training & Loss Function Changes

**Original:**

* `CTCLoss` with Adam optimizer.
* Learning rate scheduler: exponential decay after 30 epochs.

**Modified:**

* **Custom loss and evaluation callbacks**:

  * `CTCLoss` returns mean over batch.
  * `DecodeAndEvaluate` computes CER (character error rate) and WER (word error rate) at epoch end.
  * `ProduceExample` callback preserved for qualitative checks.

* **CosineAnnealingScheduler** replaces original LR scheduler:

```python
class CosineAnnealingScheduler(tf.keras.callbacks.Callback):
    def __init__(self, max_lr, min_lr, total_epochs, warmup_epochs=5):
        super().__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.max_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        print(f"\nEpoch {epoch+1}: Learning rate set to {lr:.6f}")
```

* Training example:

```python
model.fit(
    train,
    validation_data=test.batch(1),
    epochs=4,
    callbacks=[checkpoint_callback, schedule_callback, example_callback]
)
```

---

## 4. Other Enhancements

* GIF visualization and single-frame preview scaled to 0–255 for correct rendering.
* Improved dataset splitting and batching for faster training.
* Full GPU/CPU memory optimizations to handle larger models efficiently.

---

## 5. Summary of Improvements

| Feature                 | Original                          | Modified                                    |
| ----------------------- | --------------------------------- | ------------------------------------------- |
| Data Loading            | On-the-fly video decoding         | Preprocessed `.npz` batches                 |
| CNN Filters             | `[128, 256, 75]`                  | `[32, 64, 128]`                             |
| BiLSTM Layers           | 128 units                         | 128 units using `RNN(LSTMCell(...))` on CPU |
| Dropout                 | 0.5                               | 0.3                                         |
| Residual Connections    | None                              | Added to 3D-CNN blocks                      |
| Normalization           | BatchNorm                         | BatchNorm + LayerNorm                       |
| LR Scheduler            | Exponential decay after 30 epochs | Cosine annealing with warmup                |
| Evaluation Metrics      | None                              | CER + WER via `DecodeAndEvaluate`           |
| Training Time per Epoch | ~5 hours                          | ~12 minutes                                 |

---

This version improves training efficiency, reduces memory usage, adds robust evaluation metrics, and maintains high model performance.
