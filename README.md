# nlp-transformers-journey

# Sentiment-BERT 

[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Sentiment--BERT-blue?logo=huggingface)](https://huggingface.co/Kash123aa/sentiment-bert)

Try the model on Hugging Face 👉 [Sentiment-BERT on Hugging Face](https://huggingface.co/Kash123aa/sentiment-bert)


A **fine-tuned BERT** model for sentiment analysis on the SST-2 dataset : classifies sentences as **Positive** or **Negative**.  
Built with 🤗 Hugging Face Transformers and trained using transfer learning.

---

## 🚀 Features

- Based on [bert-base-uncased](https://huggingface.co/bert-base-uncased)  
- Fine-tuned on [SST-2 dataset](https://huggingface.co/datasets/SetFit/sst2)  
- Fast, accurate sentiment classification  
- Easy to use with Hugging Face's `transformers` pipeline  

---

## 📊 Model Performance

| Metric        | Score  |
|---------------|--------|
| Accuracy      | 90.6%  |
| Validation Loss | 0.51  |

---

## 🛠️ How to Use

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="Kash123aa/sentiment-bert")

result = classifier("I am so excited to attend the Karan Aujla's concert! ")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]
```
---

## Model description

This model adds a classification head (dense layer) on top of the pre-trained BERT base and is fine-tuned using supervised labels from the SST-2 sentiment dataset. The pre-trained weights are retained for their general language understanding, and only the classification head is newly initialized.


## Intended uses:

- Sentiment classification of movie reviews, social media posts, or similar short texts.
- ducational purposes for understanding transfer learning using Hugging Face Transformers.

## Limitations:

- Trained on SST2, so performance on out-of-domain or multilingual data may be limited.
- Only handles binary sentiment (positive/negative); it won’t generalize well to multi-class emotion classification.

## Training and evaluation data

- Dataset: SetFit/sst2
- Train size: 6,920 samples
- Validation size: 872 samples
- Test size: 1,821 samples
- Labels: 0 = negative, 1 = positive

## Training procedure
The model was fine-tuned by adding a classification layer on top of the pre-trained BERT base. The SST-2 sentiment dataset was tokenized and padded. Training was done using the Hugging Face Trainer for 3 epochs with evaluation after each epoch. Accuracy was used as the evaluation metric. Finally, the model was tested on a separate test set.

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 3.0


### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.4123        | 1.0   | 865  | 0.3335          | 0.9163   |
| 0.1823        | 2.0   | 1730 | 0.3848          | 0.9083   |
| 0.0458        | 3.0   | 2595 | 0.5594          | 0.9071   |


### Framework versions

- Transformers 4.53.2
- Pytorch 2.6.0+cu124
- Datasets 4.0.0
- Tokenizers 0.21.2

