import React, { useEffect, useRef } from "react"; // Import useEffect and useRef
import ChatBubble from "./ChatBubble";

export default function ChatBox({ messages }) {
    const chatBoxRef = useRef(null); // Create a ref for the chat box container

    // useEffect hook to scroll to the bottom whenever messages change
    useEffect(() => {
        if (chatBoxRef.current) {
            chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
        }
    }, [messages]); // Dependency array: runs when 'messages' array updates

    return (
        <div
            ref={chatBoxRef} // Attach the ref to the scrollable container
            className="flex-1 overflow-y-auto px-4 py-2 space-y-4"
        >
            {messages.map((msg, i) => (
                <ChatBubble key={i} role={msg.role} content={msg.content} image={msg.image} />
            ))}

        </div>
    );
}
