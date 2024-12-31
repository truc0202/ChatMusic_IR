from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import secrets
from datetime import datetime
from intent_classification import IntentClassifier
# from predict_model import predict_sentiment
from music_bot import get_music_recommendation,get_sentiment_recommendation,get_similarity_song  # Module gợi ý bài hát
from recommend_similarity_song import is_related_query

app = Flask(__name__)
CORS(app)

# Khởi tạo IntentClassifier
intent_classifier = IntentClassifier()

# Lấy thời gian hiện tại
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Route chính để hiển thị giao diện
@app.route("/", methods=["GET"])
def frontend_message():
    return render_template("index.html")

# Route API để xử lý tin nhắn người dùng
@app.route("/get_response", methods=["GET", "POST"])
def get_response():
    if request.method == "GET":
        receive_time = get_current_time()
        return jsonify({
            'token': secrets.token_hex(),
            'messages': [
                {
                    'username': 'CHATBOT',
                    'timestamp': receive_time,
                    'message': (
                        "Xin chào! Mình là Chatbot thông minh vjp pro có chức năng gợi ý bài hát. "
                        "Bạn hãy nhập câu hỏi vào khung chat bên dưới để bắt đầu giao tiếp nhé!"
                    ),
                },
            ],
        })

    if request.method == "POST":
        if not request.json or "message" not in request.json:
            return jsonify({"response": "Dữ liệu không hợp lệ. Vui lòng gửi một tin nhắn hợp lệ."})

        user_message = request.json.get("message", "").strip()
        if not user_message:
            return jsonify({"response": "Bạn hãy nhập một tin nhắn trước nhé!"})

        try:
            # Sử dụng IntentClassifier để lấy phản hồi
            response = intent_classifier.get_response(user_message)
            if response:
                return jsonify({"response": response})

            if is_related_query(user_message):
                if get_similarity_song():
                    return jsonify({"response": get_similarity_song()})
                else :
                    return jsonify({"response": "Bạn có thể đề xuất tâm trạng một bài nhạc cụ thể để tôi gợi ý cho bạn?"})
            # Kiểm tra xem user_message có chứa các từ liên quan đến việc gợi ý bài hát hay không
            music_keywords = ["nhạc", "bài hát", "nghe", "bài"]
            if any(keyword in user_message.lower() for keyword in music_keywords):

                # Gợi ý bài hát dựa trên cảm xúc
                music_recommendation = get_music_recommendation(user_message)

                if music_recommendation:
                    return jsonify({"response": music_recommendation})

            return jsonify({"response": get_sentiment_recommendation(user_message)})
        except Exception as e:
            return jsonify({"response": f"Đã xảy ra lỗi: {str(e)}"})
if __name__ == "__main__":
    app.run(debug=True)