from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import secrets
from datetime import datetime
import random
import nltk
from nltk.chat.util import Chat, reflections
# from predict_model import predict_sentiment
from music_bot import get_music_recommendation,get_sentiment_recommendation,get_similarity_song  # Module gợi ý bài hát
from recommend_similarity_song import is_related_query
app = Flask(__name__)
CORS(app)

# Mẫu câu hỏi và trả lời cho chatbot

pairs = [
    [r"(.*) chào|xin chào|chào bạn|hi|hello|hey|lô|zô", 
     ["Chào bạn! Bạn khỏe không? Mình có thể giúp gì hôm nay?", 
      "Xin chào! Rất vui được trò chuyện cùng bạn.", 
      "Chào bạn! Có chuyện gì vui chia sẻ với mình không?"]],
      
    [r"bạn khỏe không|khỏe không|bạn có khỏe không|how are you", 
     ["Mình là chatbot, luôn sẵn sàng hỗ trợ bạn! Cảm ơn bạn đã hỏi nhé!", 
      "Mình vẫn khỏe, cảm ơn bạn! Bạn cần hỗ trợ gì không?", 
      "Mình ổn, cảm ơn bạn! Bạn thì sao?"]],
      
    [r"(.*) tên gì|là gì|(.*) gọi là gì|name|(.*) giới thiệu",
     ["Mình là MusicBot, trợ lý chatbot của bạn! Rất vui được gặp bạn."]],
      
    [r"(.*) thời tiết|thời tiết hôm nay|(.*) thời tiết như thế nào|weather",
     ["Mình không thể kiểm tra thời tiết trực tiếp, nhưng bạn có thể tra cứu trên Google hoặc các ứng dụng thời tiết nhé!",
      "Mình không biết chính xác thời tiết hiện tại, bạn có thể xem trên internet xem sao.",
      "Thời tiết thay đổi liên tục, bạn hãy xem dự báo thời tiết trên các kênh chính thống nhé!"]],
      
    [r"tạm biệt|bye|goodbye|bai|bái bai|pp",
     ["Tạm biệt! Chúc bạn một ngày tuyệt vời!",
      "Hẹn gặp lại bạn sau nhé! Chúc bạn vui vẻ!",
      "Rất vui được trò chuyện với bạn. Tạm biệt nhé!"]],
      
    [r"cảm ơn|cảm ơn bạn|cảm ơn nhé|thanks|thank you",
     ["Không có chi! Mình luôn ở đây khi bạn cần.",
      "Rất vui được giúp đỡ bạn. Nếu cần gì thêm, đừng ngại nhé!",
      "Cảm ơn bạn đã tin tưởng mình!"]],
      
    [r"(.*) làm gì|làm gì bây giờ|(.*) làm được gì|có thể làm gì",
     ["Mình có thể giúp bạn tìm nhạc theo tâm trạng hiện tại. Bạn muốn nghe nhạc vui hay buồn?",
      "Hãy cho mình biết tâm trạng của bạn, mình sẽ tìm bài hát phù hợp cho bạn!",
      "Mình ở đây để giúp bạn tìm nhạc! Bạn muốn nghe gì hôm nay?"]]
]


# Tạo chatbot với các mẫu câu
chatbot = Chat(pairs, reflections)

# Lấy thời gian hiện tại
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Route chính để hiển thị giao diện
@app.route("/", methods=["GET"])

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
            # Trả lời qua NLTK Chat nếu có mẫu câu phù hợp
            response = chatbot.respond(user_message.lower())
            if response:
                return jsonify({"response": response})
            if is_related_query(user_message):
                return jsonify({"response": get_similarity_song(user_message)})
            # Kiểm tra xem user_message có chứa các từ liên quan đến việc gợi ý bài hát hay không
            music_keywords = ["nhạc", "bài hát", "nghe", "bài"]
            if any(keyword in user_message.lower() for keyword in music_keywords):

                # Gợi ý bài hát dựa trên cảm xúc
                music_recommendation = get_music_recommendation(user_message)

                if music_recommendation:
                    return jsonify({"response": music_recommendation})
                else:
                    return jsonify({"response": "Không thể gợi ý bài hát. Vui lòng thử lại sau."})

            return jsonify({"response": get_sentiment_recommendation(user_message)})
        except Exception as e:
            return jsonify({"response": f"Đã xảy ra lỗi: {str(e)}"})
if __name__ == "__main__":
    app.run(debug=True)
