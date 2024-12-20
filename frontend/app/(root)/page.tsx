import Chat from "@/components/Chat";
import Header from "@/components/Header";
import React from "react";

const Page = () => {
  return (
    <div className="w-full h-screen bg-primary-900">
      <div className="h-[12%]">
        <Header />
      </div>
      <div className="h-[88%] flex justify-center items-center">
        <Chat />
      </div>
    </div>
  );
};

export default Page;
