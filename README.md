# nlp-transformers-journey

# Sentiment-BERT 

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
