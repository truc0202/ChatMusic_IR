from nltk.chat.util import Chat, reflections
# Mẫu câu hỏi và trả lời cho chatbot
pairs = [
    [r"(.*) chào|xin chào|chào bạn|hi|hello|hey|lô|zô", 
     ["Chào bạn! Rất vui được trò chuyện cùng bạn. Có chuyện gì chia sẻ với mình không?"]],
      
    [r"(.*) khỏe không|(.*) thế nào|(.*) sao|(.*) ổn không|(.*) tình hình", 
     ["Mình là chatbot, luôn sẵn sàng hỗ trợ bạn! Cảm ơn bạn đã hỏi nhé!", 
      "Mình vẫn khỏe, cảm ơn bạn! Bạn cần hỗ trợ gì không?", 
      "Mình ổn, cảm ơn bạn! Bạn thì sao?"]],
      
    [r"(.*) tên gì|(.*) là gì|(.*) giới thiệu",
     ["Mình là MusicBot, trợ lý chatbot của bạn! Rất vui được gặp bạn."]],
      
    [r"(.*) tạm biệt|tạm biệt|bye|goodbye|bai|bái bai|pp",
     ["Tạm biệt! Chúc bạn một ngày tuyệt vời!",
      "Hẹn gặp lại bạn sau nhé! Chúc bạn vui vẻ!",
      "Rất vui được trò chuyện với bạn. Tạm biệt nhé!"]],
      
    [r"(.*) cảm ơn|cảm ơn|thanks|thank you", # Nếu chuỗi người dùng gửi không khớp với regex (chẳng hạn có ký tự thừa hoặc thiếu), chatbot sẽ không nhận diện đúng.
     ["Không có chi! Mình luôn ở đây khi bạn cần.",
      "Rất vui được giúp đỡ bạn. Nếu cần gì thêm, đừng ngại nhé!",
      "Cảm ơn bạn đã tin tưởng mình!"]],
      
    [r"(.*) làm gì|(.*) làm được gì",
     ["Mình có thể giúp bạn tìm nhạc theo tâm trạng hiện tại. Bạn muốn nghe nhạc vui hay buồn?",
      "Hãy cho mình biết tâm trạng của bạn, mình sẽ tìm bài hát phù hợp cho bạn!",
      "Mình ở đây để giúp bạn tìm nhạc! Bạn muốn nghe gì hôm nay?"]]
]
# Tạo một lớp chatbot với các mẫu câu trên
class IntentClassifier:
    def __init__(self):
        self.chatbot = Chat(pairs, reflections)

    def get_response(self, user_message: str) -> str:
        """Nhận diện ý định và trả về phản hồi phù hợp."""
        return self.chatbot.respond(user_message.lower())