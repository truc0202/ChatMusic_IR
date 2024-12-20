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
    "Nữa đi", "Thêm bài nữa", "thêm 1 bài nữa", "thêm 1 bài đi", "tiếp tục nào",
    "thêm 1 bài nữa nào", "1 bài nữa", "1 bài nữa nào", "một bài nữa", "one more",
    "tiếp", "bài tiếp theo", "Cho mình nghe thêm 1 bài nữa nhé", "cho mình nghe thêm 1 bài",
    "bot ơi cho mình nghe thêm đi", "bot ngu thêm 1 bài nữa đi", "thêm 1 bài nữa đi bot",
    "1 more", "Nữa", "tui muốn nghe nữa", "nữa đi nữa đi", "tôi muốn nghe thêm 1 bài",
    "một bài nữa thì sao", "tiếp bài nữa", "hãy cho mình một bài nữa đi", 
    "cho xin thêm một bài nữa", "xin thêm bài nữa", "nữa nào", "bài nữa nào", "bài nữa đi bot"
]



def is_related_query(user_query):
    """
    Kiểm tra xem câu query có liên quan đến yêu cầu 'thêm bài hát' không.
    """
    sw = []
    with open("./dataset/data_model/vietnamese-stopwords.txt", encoding='utf-8') as f:
        stopwords = f.readlines()
    stopwords = [preprocess_text(word.strip()) for word in stopwords]

    # Vector hóa câu query của người dùng
    # Chuẩn bị vectorizer
    vectorizer = TfidfVectorizer(stop_words=stopwords)#stop_words=stopwords)
    query_matrix = vectorizer.fit_transform(sample_queries)
    user_vector = vectorizer.transform([user_query])
    
    # Tính độ tương đồng cosine
    similarity_scores = cosine_similarity(user_vector, query_matrix)
    
    # Ngưỡng quyết định (ví dụ: 0.5)
    if similarity_scores.max() > 0.5:
        return True
    return False


# Hàm tính độ tương đồng và đề xuất bài hát
def recommend_song(input_text, data, top_n=1):
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
    
    # Trả về kết quả
    recommendations = data.iloc[top_indices]
    return recommendations['song_name'].tolist(), recommendations['artist_name'].tolist(), recommendations['song_URL'].tolist()

# Ví dụ sử dụng hàm recommend_song
if __name__ == "__main__":
    # Đọc dữ liệu từ tệp CSV
    csv_path = "./dataset/Data_music/Combined_songs.csv"
    df = pd.read_csv(csv_path, encoding='utf-8')

    input_text = "Xa em"
    song_names, artist_names, song_urls = recommend_song(input_text, df)
    print(f"Gợi ý bài hát cho '{input_text}':")
    for song_name, artist_name, song_url in zip(song_names, artist_names, song_urls):
        print(f"Bài hát: {song_name}, Nghệ sĩ: {artist_name}, URL: {song_url}")