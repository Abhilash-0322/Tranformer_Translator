# Transformer-Based Language Translator (From Scratch)

This project is a **Transformer model built from scratch using PyTorch** to translate text from one language to another. It is trained on the **OPUS Books dataset** from Hugging Face. I implemented this project following a tutorial by **Umar Jamil**, but this README reflects my own understanding and modifications.

---

## 🔥 Features
- **Custom-built Transformer** (encoder-decoder architecture)
- **Trained on OPUS Books dataset** (a multilingual corpus)
- **Implements Attention Mechanisms** (Self-Attention & Cross-Attention)
- **Uses PyTorch for training & inference**
- **Tokenization & Data Preprocessing with Hugging Face**
- **Custom Training Loop with PyTorch Lightning (optional)**

---

## 📌 Dataset: OPUS Books
The model is trained on the [OPUS Books dataset](https://huggingface.co/datasets/opus_books), a collection of literary texts available in multiple languages. This dataset provides high-quality parallel translations, making it ideal for training translation models.

---

## 🛠️ Setup & Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Abhilash-0322/Tranformer_Translator.git
cd Tranformer_Translator
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
Ensure you have **PyTorch**, **Hugging Face Transformers**, and **Tokenizers** installed.

### 3️⃣ Download Dataset
```python
from datasets import load_dataset

dataset = load_dataset("opus_books", "en-fr")  # Example: English to French
```
Modify the language pair (`"en-fr"`) as needed.

---

## 🚀 Model Architecture
This project implements the **original Transformer architecture** from the "Attention is All You Need" paper. The key components include:

- **Positional Encoding**: Adds order information to word embeddings.
- **Multi-Head Self-Attention**: Captures relationships between words.
- **Feed-Forward Networks**: Provides non-linearity & depth.
- **Layer Normalization & Dropout**: Improves stability and generalization.

---

## 🎯 Training the Model
To train the model, run:
```bash
python train.py
```
Modify `train.py` to adjust hyperparameters like learning rate, optimizer, or scheduler.

---

## 📝 Inference: Translating Text
After training, you can test translation using:
```python
from model import TransformerTranslator

model = TransformerTranslator.load_from_checkpoint("checkpoint.pth")
text = "Hello, how are you?"
predicted_translation = model.translate(text)
print(predicted_translation)
```

---

## 🛠️ Future Improvements
- Fine-tune with **larger datasets** (e.g., WMT).
- Integrate **BLEU score evaluation**.
- Convert to **TorchScript or ONNX** for deployment.
- Explore **beam search decoding** for better translations.

---

## ✨ Acknowledgments
Big shoutout to **Umar Jamil** for the original tutorial that laid the foundation for this project. I extended the implementation with my own **modifications, explanations, and optimizations**.

---

## 📜 License
This project is open-source under the **MIT License**.

---

## 🌟 Connect with Me
If you found this project helpful or have suggestions, reach out!
- GitHub: [Abhilash-0322](https://github.com/Abhilash-0322)
- LinkedIn: [Abhilash Maurya](https://www.linkedin.com/in/abhilash-maurya-b615b9277)

