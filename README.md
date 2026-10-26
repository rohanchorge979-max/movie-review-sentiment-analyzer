# Movie Review Sentiment Analyzer using TF-IDF and DistilBERT

An NLP mini project that reads a movie review and predicts whether it is **positive** or **negative**. It compares a classical machine learning model with a fine-tuned transformer and includes a live web demo.

## Overview
- **Task:** Binary sentiment classification of movie reviews
- **Dataset:** IMDB movie reviews (4,000 for training, 1,000 for testing)
- **Models compared:**
  1. TF-IDF + Logistic Regression (baseline)
  2. DistilBERT, fine-tuned for 1 epoch
- **Demo:** Gradio web app that returns the sentiment with a confidence score

## How it works
1. Load the IMDB dataset from Hugging Face
2. Clean the text (lowercase, remove HTML tags and punctuation)
3. Train the TF-IDF + Logistic Regression baseline
4. Fine-tune DistilBERT on the same reviews
5. Evaluate both models with accuracy, F1 score and confusion matrices
6. Launch the Gradio app to test any review

## Results

| Model | Accuracy | F1 Score |
|---|---|---|
| TF-IDF + Logistic Regression | [your value] | [your value] |
| DistilBERT (fine-tuned) | [your value] | [your value] |

## Tools and technologies
Python, Google Colab (T4 GPU), Hugging Face Datasets and Transformers, PyTorch, scikit-learn, Pandas, Matplotlib, Gradio

## How to run
1. Open the notebook file in this repository with Google Colab
2. Set **Runtime → Change runtime type → T4 GPU**
3. Click **Runtime → Run all**
4. Open the Gradio link printed at the end and type a review

## Screenshots
[Add a screenshot of the Gradio app here]

## Future scope
- Train on the full 25,000 reviews
- Support Hindi, Marathi and Hinglish reviews
- Explain predictions with LIME or SHAP
- Host the app permanently on Hugging Face Spaces

