import Image from "next/image";
import React from "react";

const Header = () => {
  return (
    <div className="flex justify-between items-center px-16 w-full h-full bg-primary-700 rounded-b-2xl">
      <h1 className="font-bold text-[18px] text-primary-500">ChatMusic</h1>
      <Image
        src={"./assets/images/textLogo.svg"}
        alt="Logo spotify"
        width={147}
        height={21}
      />
    </div>
  );
};

export default Header;
