# DistilBERT Fine-Tuned on SQuAD

A comprehensive implementation of fine-tuning DistilBERT for Question Answering on the Stanford Question Answering Dataset (SQuAD).

## Overview

This project demonstrates how to fine-tune the DistilBERT model for extractive question answering tasks using the SQuAD dataset. The implementation includes complete data preprocessing, model training, and evaluation pipelines using the Hugging Face Transformers library.

## Features

- **Efficient Model Architecture**: Uses DistilBERT, a distilled version of BERT that is 40% smaller and 60% faster while retaining 97% of BERT's language understanding capabilities
- **Complete Training Pipeline**: End-to-end implementation from data loading to model evaluation
- **Advanced Tokenization**: Handles long contexts with stride-based chunking and offset mapping
- **Robust Evaluation**: Implements SQuAD metrics (Exact Match and F1 Score) with post-processing
- **TPU/GPU Support**: Optimized for accelerated training on Google Colab

## Model Performance

The fine-tuned model achieves the following results on the SQuAD validation set:

- **Exact Match**: 77.32%
- **F1 Score**: 85.49%

## Requirements

```
transformers
datasets
torch
numpy
```

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install transformers datasets torch numpy
```

## Dataset

This project uses the **SQuAD (Stanford Question Answering Dataset)**, which contains:
- **Training Set**: ~87,000 question-answer pairs
- **Validation Set**: ~10,000 question-answer pairs

The dataset is automatically downloaded from Hugging Face Datasets.

## Model Architecture

**Base Model**: `distilbert-base-uncased`

The model uses `AutoModelForQuestionAnswering`, which adds a linear layer on top of DistilBERT outputs to predict:
- **Start logits**: Probability distribution over tokens for answer start position
- **End logits**: Probability distribution over tokens for answer end position

## Training Configuration

### Tokenization Parameters
- **Max Length**: 384 tokens
- **Stride**: 128 tokens (for handling long contexts)
- **Truncation**: Only the context is truncated, questions are preserved
- **Padding**: Max length padding

### Training Arguments
- **Learning Rate**: 2e-5
- **Batch Size**: 16 (training), 8 (evaluation)
- **Epochs**: 3
- **Weight Decay**: 0.01
- **Warmup Steps**: 500
- **Evaluation Strategy**: Per epoch

## Usage

### Running the Notebook

1. Open the notebook in Google Colab or Jupyter:
   ```
   DistilBERT_Fine_Tuned_on_SQuAD.ipynb
   ```

2. Select a runtime with GPU/TPU acceleration (recommended)

3. Run all cells sequentially

### Key Components

#### 1. Data Loading
```python
from datasets import load_dataset
dataset = load_dataset("squad")
```

#### 2. Tokenization
The preprocessing function handles:
- Tokenizing questions and contexts
- Managing overflow tokens for long contexts
- Mapping character positions to token positions
- Labeling start and end positions for answers

#### 3. Model Training
Uses Hugging Face `Trainer` API with custom training arguments for efficient fine-tuning.

#### 4. Evaluation
Implements post-processing to:
- Convert logits to answer predictions
- Handle impossible answers
- Select best answers based on combined start/end scores
- Calculate SQuAD metrics (Exact Match and F1)

## Project Structure

```
.
├── DistilBERT_Fine_Tuned_on_SQuAD.ipynb  # Main notebook
└── README.md                              # This file
```

## How It Works

### 1. Preprocessing
- Questions and contexts are tokenized together with special tokens
- Long contexts are split into overlapping chunks (stride=128)
- Answer positions are mapped from character indices to token indices

### 2. Training
- The model learns to predict the start and end token positions of answers
- Uses cross-entropy loss for both start and end predictions
- Optimized with AdamW optimizer and linear learning rate schedule

### 3. Inference
- Model outputs start and end logits for each token
- Post-processing selects the best answer span based on:
  - Combined start + end scores
  - Valid answer constraints (start ≤ end, within context, reasonable length)
  - N-best answer candidates

## Evaluation Metrics

### Exact Match (EM)
Percentage of predictions that match the ground truth answer exactly (after normalization).

### F1 Score
Measures the average overlap between predicted and ground truth answers at the token level.

## Hardware Requirements

- **Minimum**: CPU with 8GB RAM (slow training)
- **Recommended**: GPU with 12GB+ VRAM or TPU
- **Optimal**: Google Colab with TPU runtime

## Training Time

- **TPU**: ~30-45 minutes for 3 epochs
- **GPU (T4)**: ~1-2 hours for 3 epochs
- **CPU**: Several hours (not recommended)

## Customization

### Adjusting Hyperparameters

Modify the `TrainingArguments` in the notebook:

```python
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,  # Adjust learning rate
    per_device_train_batch_size=16,  # Adjust batch size
    num_train_epochs=3,  # Change number of epochs
    # ... other parameters
)
```

### Using Different Models

Replace the model name:

```python
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

## Troubleshooting

### Out of Memory Errors
- Reduce `per_device_train_batch_size`
- Reduce `max_length` in tokenization
- Use gradient accumulation

### Slow Training
- Enable GPU/TPU acceleration
- Increase batch size if memory allows
- Use mixed precision training (fp16)

## References

- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Question Answering Guide](https://huggingface.co/docs/transformers/tasks/question_answering)

## License

This project is for educational purposes. Please refer to the licenses of the underlying models and datasets:
- DistilBERT: Apache 2.0
- SQuAD Dataset: CC BY-SA 4.0

## Acknowledgments

- Hugging Face for the Transformers library and model hosting
- Stanford NLP for the SQuAD dataset
- Google for Colab and TPU resources

## Author

Created as a demonstration of fine-tuning transformer models for question answering tasks.

---

**Note**: This notebook is designed to run in Google Colab with TPU/GPU acceleration for optimal performance.
