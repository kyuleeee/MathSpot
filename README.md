# MiBERT
This is a repository for identifying spoken math in sentence 

# MathBridge Adapter Training

This repository contains a script for training a BERT-based adapter model for recognizing mathematical expressions in spoken English using the MathBridge dataset.

## Features
- Uses **AdapterFusion** techniques with different adapter types:
  - **Pfeiffer** (`seq_bn`)
  - **Houlsby** (`double_seq_bn`)
  - **Parallel** (`par_seq_bn`)
- Token classification task (Named Entity Recognition, NER) with labels:
  - `O`: Outside mathematical expressions
  - `B-MATH`: Beginning of a math expression
  - `I-MATH`: Inside a math expression
- Supports training, validation, and testing
- Saves trained adapter and tokenizer
- Optionally uploads the model and adapter to Hugging Face Hub

## Installation
Ensure you have Python installed and the necessary dependencies:

```sh
pip install torch transformers adapter-transformers datasets seqeval
```

## Usage
Run the script with default parameters:

```sh
python train_adapter.py
```

### Arguments
The script supports multiple arguments to customize training:

| Argument | Description | Default |
|----------|-------------|---------|
| `--seed` | Random seed | 42 |
| `--batch_size` | Batch size | 32 |
| `--learning_rate` | Learning rate | 1e-4 |
| `--weight_decay` | Weight decay | 0.01 |
| `--max_seq_length` | Maximum sequence length | 128 |
| `--num_train_epochs` | Number of training epochs | 1 |
| `--adapter_type` | Adapter type (`seq`, `double`, `par`) | `seq` |
| `--reduction_factor` | Adapter reduction factor | 16 |

Example with a different adapter type:

```sh
python train_adapter.py --adapter_type double --num_train_epochs 3
```

## Dataset
The script uses the **MathBridge dataset** from Hugging Face:

```python
load_dataset('Kyudan/MathBridge')
```

This dataset provides contextual spoken English sentences with math expressions labeled accordingly.

## Model Training
- The script tokenizes input texts and assigns labels based on the position of mathematical expressions.
- It trains a **BERT-based adapter model** using **AdapterFusion**.
- Training progress and evaluation metrics (precision, recall, F1-score) are logged.

## Evaluation
After training, the model is evaluated using:

- **Accuracy**
- **Precision, Recall, and F1-score** (via `seqeval`)

## Saving & Uploading
The trained model and adapter are saved locally:

```sh
mathbridge-bert-adapter-seq/
mathbridge-bert-adapter-seq_adapter/
```

Optionally, the model and adapter can be uploaded to **Hugging Face Hub**:

```python
model.push_to_hub(model_name)
model.push_adapter_to_hub("mathbridge_adapter", "mathbridge_adapter", model_name)
```

## Acknowledgments
This work is based on the **AdapterFusion** framework and **Hugging Face Transformers**.


