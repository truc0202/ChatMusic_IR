import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ dấu câu
    text = text.lower()  # Chuyển thành chữ thường
    return text

sample_queries = [
    # Cách diễn đạt phổ thông
    "nữa đi", "cho thêm bài nữa", "thêm 1 bài nữa", "nghe thêm đi", "bài nữa nào",
    "một bài nữa", "thêm bài nữa nào", "nghe bài khác đi", "cho thêm bài hát", "nghe tiếp đi",
    "tiếp tục", "thêm 1 bài", "bài tiếp theo", "nghe thêm 1 bài đi", "cho xin bài khác"
]




def is_related_query(user_query):
    """
    Kiểm tra xem câu query có liên quan đến yêu cầu 'thêm bài hát' không.
    """
    # Vector hóa câu query của người dùng
    # Chuẩn bị vectorizer
    vectorizer = TfidfVectorizer()
    query_matrix = vectorizer.fit_transform(sample_queries)
    user_query = preprocess_text(user_query)
    user_vector = vectorizer.transform([user_query])
    
    # Tính độ tương đồng cosine
    similarity_scores = cosine_similarity(user_vector, query_matrix)
    print(similarity_scores)
    # # # Ngưỡng quyết định (ví dụ: 0.8)
    if similarity_scores.max() > 0.8:
        return True
    return False



# Hàm tính độ tương đồng và đề xuất bài hát
def recommend_song(input_text, data, top_n=5):
    if input_text == "":
        return None, None, None
    # Tạo cột tổng hợp dữ liệu (kết hợp tiêu đề và ca sĩ)
    data['combined'] = data['song_name'] + " " + data['artist_name']
    input_text = preprocess_text(input_text)

    # Vector hóa dữ liệu bằng TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['combined'])

    # Vector hóa input
    input_vector = vectorizer.transform([input_text])
    
    # Tính độ tương đồng cosine
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix)
    
    # Lấy top N bài hát tương tự
    top_indices = similarity_scores[0].argsort()[-top_n:][::-1]
    
    # Trả về kết quả của bài hát thứ 2
    recommendations = data.iloc[top_indices]
    if len(recommendations) > 1:
        second_recommendation = recommendations.iloc[1]
        return second_recommendation['song_name'], second_recommendation['artist_name'], second_recommendation['song_URL']
    else:
        return None, None, None

# Ví dụ sử dụng hàm recommend_song
if __name__ == "__main__":
    # Đọc dữ liệu từ tệp CSV
    # csv_path = "c:/Users/Truc/Desktop/HK1 2024-2025/Truy vấn thông tin đa phương tiện CS336/Đồ án chatbot/official_code/chatbot/dataset/Data_music/Combined_songs.csv"
    # df = pd.read_csv(csv_path, encoding='utf-8')
    # input_text = " 'Gác Lại Âu Lo' của Da LAB"
    # song_name, artist_name, song_url = recommend_song(input_text, df)
    # if song_name and artist_name and song_url:
    #     print(f"Gợi ý bài hát cho '{input_text}':")
    #     print(f"Bài hát: {song_name}, Nghệ sĩ: {artist_name}, URL: {song_url}")
    # else:
    #     print("Không có đủ bài hát tương tự để gợi ý.")
    print(is_related_query("hôm nay tôi khá là vui"))