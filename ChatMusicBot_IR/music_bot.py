import pandas as pd
import random
from predict_model import predict_sentiment
from recommend_similarity_song import recommend_song

# Đọc dữ liệu từ tệp CSV
csv_path = "./dataset/Data_music/Combined_songs.csv"
df = pd.read_csv(csv_path, encoding='utf-8')


# Chuyển đổi dữ liệu thành cấu trúc phù hợp
# music_data = {}
# for label in df['label'].unique():
#     music_data[label] = df[df['label'] == label].to_dict(orient='records')


# Sentiment recommendation logic
def get_sentiment_recommendation(user_input):
    sentiment = predict_sentiment(user_input)
    return f"Mình thấy bạn đang {sentiment}. Bạn có muốn lắng nghe những giai điệu vui tươi hay buồn bã để hòa quyện cùng cảm xúc của mình không?"

# Music recommendation logic
def get_music_recommendation(user_input):
    # Dự đoán cảm xúc từ tin nhắn
    sentiment = predict_sentiment(user_input)
    print(sentiment)
    if sentiment in df['label'].unique():
        song_info = random.choice(df[df['label'] == sentiment].to_dict(orient='records'))
        song_name = song_info['song_name']
        artist_name = song_info['artist_name']
        song_url = song_info['song_URL']
        return f"Bạn muốn nghe một bài nhạc {sentiment}. \nHãy thử nghe bài '{song_name}' của {artist_name}. \nBạn có thể nghe tại đây: \n{song_url}"
    else:
        return "Mình chưa hiểu ý bạn. Vui lòng nhập lại tâm trạng hoặc thể loại nhạc."


def get_similarity_song(user_input,data = df):
    song_title, artist, song_url = recommend_song(user_input,data)
    for song_title, artist in zip(song_title, artist):
        responses = [
            f"Đây là bài nhạc tương tự {song_title}: \ncủa {artist} \n Hãy thử nghe và cảm nhận nhé! {song_url}",
            f"Bài nhạc {song_title} của {artist} là một gợi ý tuyệt vời cho bạn.\nBạn có thể nghe tại đây: {song_url}",
            f"Mình nghĩ bạn sẽ thích bài nhạc này: {song_title} của {artist}\nBạn có thể nghe tại đây: {song_url}",
            f"Xin giới thiệu bài hát tương tự: {song_title} của {artist}\nBạn có thể nghe tại đây: {song_url}"
        ]
    return random.choice(responses)


# Main function
def main():
    
    # with open(music_data_path, "r", encoding="utf-8") as f:
    #     music_data = json.load(f)

    # Ví dụ đầu vào
    user_input = "Xa em"
    
    # Gọi hàm và in kết quả
    result = get_similarity_song(user_input)
    print(f"Gợi ý bài hát cho bạn: \n{result}")

    # # User interaction
    # print("Chào mừng bạn đến với Music Bot! Hãy cho mình biết tâm trạng hoặc sở thích âm nhạc của bạn.")
    # while True:
    #     user_input = input("Bạn đang cảm thấy thế nào? (hoặc gõ 'thoát' để dừng): ")
    #     if user_input.lower() == "thoát":
    #         print("Hẹn gặp lại bạn lần sau!")
    #         break

    #     recommendation = get_music_recommendation(user_input) # cho music_data = nhac
    #     print(recommendation)

if __name__ == "__main__":
    main()


