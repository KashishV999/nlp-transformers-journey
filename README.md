# nlp-transformers-journey

# Emotion-Classify-BERT 

[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Emotion--Classify--BERT-blue?logo=huggingface)](https://huggingface.co/Kash123aa/emotion-classify)

Try the model on Hugging Face 👉 [Emotion-Classify on Hugging Face](https://huggingface.co/Kash123aa/emotion-classify)


This model is a fine-tuned version of [bert-base-uncased](https://huggingface.co/bert-base-uncased) on the [`mteb/emotion`](https://huggingface.co/datasets/mteb/emotion) dataset.  
It is trained to classify English text into one of six emotion categories:

- sadness
- joy
- love
- anger
- fear
- surprise

---

## 🚀 Features

- Based on [bert-base-uncased](https://huggingface.co/bert-base-uncased)  
- Fine-tuned on [SST-2 dataset](https://huggingface.co/datasets/mteb/emotion)  
- Fast, accurate 6 emotion classification  
- Easy to use with Hugging Face's `transformers` pipeline  

---

## 📊 Model Performance

| Metric          | Score   |
|-----------------|---------|
| Accuracy        | 0.9401  |
| Validation Loss | 0.1660  |

---

## 🛠️ How to Use

```python
# Use a pipeline as a high-level helper
from transformers import pipeline

classify = pipeline("text-classification", model="Kash123aa/emotion-classify")

classify("I wanted to got to karan aujla's concert but i was unable to get tickets")
# [{'label': 'fear', 'score': 0.4427618682384491}]

classify("I went finally after 4 years to aujla's concert")
# [{'label': 'joy', 'score': 0.9822591543197632}]

classify("I smiled at the photo, but deep down, I felt hollow.")
# [{'label': 'sadness', 'score': 0.9996618032455444}]

classify("I booked tht ticktes for concert but at th end my brother canceelled the plan")
# [{'label': 'anger', 'score': 0.48287907242774963}]

```
---

## Model description

This model uses a pre-trained BERT encoder (`bert-base-uncased`) with a new classification head on top.  
During fine-tuning, the base model's parameters were frozen to retain its general language understanding while the classifier learned emotion-specific patterns from labeled text.

## Intended uses & limitations

This model is intended for:

- Emotion classification of English sentences
- Analyzing user sentiment in social media posts, reviews, or feedback
- Educational and research purposes

 Limitations
- May not handle sarcasm, code-switching, or very informal language well.
- Emotions are multi-dimensional; the model outputs only one top label per input.


## Training and evaluation data

The model was fine-tuned on the mteb/emotion dataset which includes 3 splits:

Train: 15,956 samples
Validation: 1,988 samples
Test: 1,986 samples
Each sample contains:

text: The input sentence
label: An integer (0–5)
label_text: Emotion name

## Training procedure

The model was fine-tuned by adding a classification layer on top of the pre-trained BERT base. The SST-2 sentiment dataset was tokenized and padded. Training was done using the Hugging Face Trainer for 3 epochs with evaluation after each epoch. Accuracy was used as the evaluation metric. Finally, the model was tested on a separate test set.

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 3.0

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.5796        | 1.0   | 998  | 0.1862          | 0.9321   |
| 0.142         | 2.0   | 1996 | 0.1386          | 0.9401   |
| 0.0895        | 3.0   | 2994 | 0.1660          | 0.9401   |


### Framework versions

- Transformers 4.52.4
- Pytorch 2.6.0+cu124
- Datasets 3.6.0
- Tokenizers 0.21.2
