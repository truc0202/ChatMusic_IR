"use client";
import { useEffect, useState, useRef } from "react";
import axios from "axios";
import Image from "next/image";

interface Message {
  username: string;
  message: string;
  timestamp: string;
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [userInput, setUserInput] = useState<string>("");
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Lấy dữ liệu ban đầu từ API
  useEffect(() => {
    const fetchMessages = async () => {
      try {
        const response = await axios.get("http://127.0.0.1:5000/get_response");
        setMessages(response.data.messages);
      } catch (error) {
        console.error("Error fetching chatbot messages:", error);
      }
    };

    fetchMessages();
  }, []);

  // Gửi tin nhắn
  const handleSendMessage = async () => {
    if (userInput.trim() !== "") {
      const userMessage: Message = {
        username: "You",
        message: userInput,
        timestamp: new Date().toLocaleTimeString(),
      };

      // Thêm tin nhắn người dùng vào danh sách
      setMessages((prev) => [...prev, userMessage]);
      setUserInput("");

      try {
        const response = await axios.post(
          "http://127.0.0.1:5000/get_response",
          {
            message: userInput,
          }
        );

        const botMessage: Message = {
          username: "Bot",
          message: response.data.response,
          timestamp: new Date().toLocaleTimeString(),
        };

        // Thêm phản hồi của bot vào danh sách
        setMessages((prev) => [...prev, botMessage]);
      } catch (error) {
        console.error("Error sending message:", error);
      }
    }
  };

  return (
    <div className="flex flex-col justify-center items-center w-[60%] h-[85%]  py-2 bg-primary-700 rounded-xl">
      <div
        className="w-[100%] px-10 outline-none border-0 h-[70%] overflow-hidden custom-scrollbar-navbar"
        id="messages"
        style={{
          overflowY: "auto",
        }}
      >
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`flex gap-4 ${
              msg.username === "You" ? "justify-end" : "justify-start"
            }`}
          >
            {msg.username === "You" ? (
              <div className="px-4 py-2 bg-[#2F2F2F]  rounded-lg inline-block">
                <p className="text-white text-[14px]">{msg.message}</p>
                <p className="text-[12px] italic text-primary-500 text-end">
                  {msg.timestamp}
                </p>
              </div>
            ) : (
              <div className="flex justify-start items-start gap-2">
                <Image
                  src="./assets/images/logo.svg"
                  height={28}
                  width={28}
                  alt="Bot Avatar"
                />
                <div>
                  <p className="text-white text-[14px]">{msg.message}</p>
                  <p className="text-[12px] italic text-primary-500">
                    {msg.timestamp}
                  </p>
                  {/* <strong>{msg.username}</strong> [{msg.timestamp}] */}
                </div>
              </div>
            )}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <div className="flex flex-row justify-between items-center w-[90%] bg-primary-600 rounded-2xl gap-4">
        <div className="w-[100%] h-[100%] py-2">
          <textarea
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
              }
            }}
            id="user-input"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder="Tin nhắn ChatMusic"
            className="font-normal text-[14px] text-primary-500 h-[60px] w-full px-4  rounded-l-2xl bg-primary-600 no-focus focus:outline-none focus:ring-0 overflow-auto custom-scrollbar-navbar resize-none"
          />
        </div>
        <button
          id="send-btn"
          onClick={handleSendMessage}
          className="mr-4 flex justify-center items-center w-[30px] h-[30px] rounded-[999px] bg-primary-400 aspect-square"
        >
          <Image
            src={"./assets/icons/up.svg"}
            alt="Icon Up"
            width={25}
            height={25}
          />
        </button>
      </div>
    </div>
  );
}
