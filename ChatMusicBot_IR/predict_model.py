# prompt: Sau khi train mô hình ở trên xong và tải xuống thì làm sao để sử dụng mô hình

import os
import torch
import numpy as np
import re
import underthesea
from transformers import AutoModel, AutoTokenizer
from sklearn.svm import SVC
from joblib import load
import json

# Load stopwords and BERT model
def load_stopwords():
    sw = []
    with open("./dataset/data_model/vietnamese-stopwords.txt", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            sw.append(line.replace("\n",""))
    return sw

def load_bert():
    v_phobert = AutoModel.from_pretrained("vinai/phobert-base")
    v_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    return v_phobert, v_tokenizer

def standardize_data(row):
    row = re.sub(r"[.,?]+$-", "", row)
    row = row.replace(",", " ").replace(".", " ").replace(";", " ").replace("“", " ").replace(":", " ").replace("”", " ").replace('"', " ").replace("'", " ").replace("!", " ").replace("?", " ").replace("-", " ").replace("?", " ")
    row = row.strip().lower()
    return row

def make_bert_features(v_text, tokenizer, phobert, sw):
    v_tokenized = []
    max_len = 100
    for i_text in v_text:
        line = underthesea.word_tokenize(i_text)
        filtered_sentence = [w for w in line if not w in sw]
        line = " ".join(filtered_sentence)
        line = underthesea.word_tokenize(line, format="text")
        line = tokenizer.encode(line)
        v_tokenized.append(line)

    padded = np.array([i + [1] * (max_len - len(i)) for i in v_tokenized])
    attention_mask = np.where(padded == 1, 0, 1)

    padded = torch.tensor(padded).to(torch.long)
    attention_mask = torch.tensor(attention_mask).to(torch.long)

    with torch.no_grad():
        last_hidden_states = phobert(input_ids=padded, attention_mask=attention_mask)

    v_features = last_hidden_states[0][:, 0, :].numpy()
    return v_features

def predict_sentiment(text):
    """Predicts the sentiment of a given text."""
    # Load the saved model and label mapping
    model = load('./model/save_model.pkl')
    label_mapping = load('./model/label_mapping.pkl')

    # Inverted label mapping for getting string labels from predictions
    inverted_label_mapping = {v: k for k, v in label_mapping.items()}
    # Load PhoBERT and tokenizer (only once)
    phobert, tokenizer = load_bert()
    phobert.eval()  # Put in evaluation mode
    sw = load_stopwords()

    # Preprocess the text
    processed_text = standardize_data(text)

    # Create BERT features
    features = make_bert_features([processed_text], tokenizer, phobert, sw)

    # Make prediction
    prediction = model.predict(features)[0]  # Get the predicted class (integer)

    # Get the string label
    sentiment = inverted_label_mapping[prediction]

    return sentiment


# # Example usage:
# text = "Tôi đang cảm thấy rất vui"
# sentiment = predict_sentiment(text)
# print(f"Sentiment: {sentiment}")

# text = "Bài hát này thật buồn"
# sentiment = predict_sentiment(text)
# print(f"Sentiment: {sentiment}")


# # Example with multiple sentences
# sentences = [
#     "Hôm nay trời đẹp quá",
#     "Bây giờ tôi đang cảm thấy tuyệt vời, bạn nghĩ tôi nên nghe nhạc gì?",
#     "Kết quả thật đáng thất vọng",
#     "Tôi đang rất buồn và cô đơn",
# ]

# for sentence in sentences:
#     sentiment = predict_sentiment(sentence)
#     print(f"Sentence: {sentence}, Sentiment: {sentiment}")