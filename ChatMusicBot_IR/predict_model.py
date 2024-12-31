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

# Hàm tải stopwordsstopwords
def load_stopwords():
    sw = []
    with open("C:/Users/Truc/Desktop/HK1 2024-2025/Truy vấn thông tin đa phương tiện CS336/Đồ án chatbot/official_code/chatbot/dataset/data_model/vietnamese-stopwords.txt", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            sw.append(line.replace("\n",""))
    return sw

# Hàm tải mô hình PhoBert và tokenizer
def load_bert():
    v_phobert = AutoModel.from_pretrained("vinai/phobert-base")
    v_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    return v_phobert, v_tokenizer

# Hàm chuẩn hóa dữ liệu
def standardize_data(row):
    row = re.sub(r"[.,?]+$-", "", row)
    row = row.replace(",", " ").replace(".", " ").replace(";", " ").replace("“", " ").replace(":", " ").replace("”", " ").replace('"', " ").replace("'", " ").replace("!", " ").replace("?", " ").replace("-", " ").replace("?", " ")
    row = row.strip().lower()
    return row

# Hàm tạo đặc trưng BERT
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

# Hàm dự đoán cảm xúc
def predict_sentiment(text):
    """Dự đoán cảm xúc của một đoạn văn bản."""
    # Tải mô hình đã lưu và ánh xạ nhãn 
    model = load('./ChatMusicBot_IR/model/save_model.pkl')
    label_mapping = load('./ChatMusicBot_IR/model/label_mapping.pkl')

    # Ánh xạ ngược để lấy nhãn dạng chuỗi từ dự đoán
    inverted_label_mapping = {v: k for k, v in label_mapping.items()}
    # Tải PhoBert và tokenizer (chỉ một lần)
    phobert, tokenizer = load_bert()
    phobert.eval()  # Đặt chế độ đánh giá
    sw = load_stopwords()

    # Tiền xử lý văn bản 
    processed_text = standardize_data(text)

    # Tạo đặc trưng BERT 
    features = make_bert_features([processed_text], tokenizer, phobert, sw)

    # Dự đoán 
    probabilities = model.predict_proba(features)[0]  # Xác suất dự đoán
    prediction = model.predict(features)[0]  # Lấy nhãn dự đoán (số nguyên)

    # Ánh xạ nhãn dạng chuỗi
    sentiment = inverted_label_mapping[prediction]

    # Lấy xác suất cao nhất
    max_prob = max(probabilities)
    print(f"Sentiment: {sentiment}, Confidence: {max_prob}")
    if max_prob > 0.7:
        return sentiment
    else:
        return None # Nếu xác suất không đạt ngưỡng, trả về None

if __name__=="__main__":
    # Example usage:
    text = "bình thường thôi" # <0.7 
    sentiment,confidence = predict_sentiment(text)
    print(f"Sentiment: {sentiment}, Confidence: {confidence}")

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