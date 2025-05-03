import React, { useState } from "react";

export default function InputBar({ onSend, setImage, image }) {
    const [input, setInput] = useState("");

    const handleSubmit = () => {
        if (input.trim() || image) {
            onSend(input);
            setInput("");
            setImage(null);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
        // Shift+Enter will now trigger the textarea's default behavior: adding a newline.
        // We don't need to do anything extra here for that.
    };

    const handleImageChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setImage(file);
            // Optional: Clear text input when an image is selected, or keep it
            // setInput("");
        }
    };

    const handleRemoveImage = () => {
        setImage(null);
    };


    return (
        <div className="p-3 border-t flex items-end gap-5 bg-white dark:bg-gray-800">
            {/* Image Upload */}
            <label className="flex flex-col items-center text-sm text-gray-600 dark:text-gray-400">
                <div className="cursor-pointer bg-gray-200 dark:bg-gray-700 px-2 py-1 rounded hover:bg-gray-300 dark:hover:bg-gray-600">
                    📷 Upload Image
                </div>
                {image && (
                    <div className="mt-1 text-xs text-center max-w-[120px] truncate flex items-center gap-1">
                        <span>{image.name}</span>
                        <button
                            onClick={(e) => { e.preventDefault(); handleRemoveImage(); }}
                            className="text-red-500 hover:text-red-700"
                            aria-label="Remove image"
                        >
                            &times;
                        </button>
                    </div>
                )}
                <input
                    type="file"
                    name="image_file"
                    accept="image/*"
                    onChange={handleImageChange}
                    className="hidden"
                />
            </label>

            {/* Text Input - textarea with native resize */}
            <textarea
                className="flex-1 p-2 border rounded bg-white text-black dark:bg-gray-800 dark:text-white"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type a message..."
                rows={3} // Set initial number of rows (adjust as needed)
                style={{ resize: 'vertical', minHeight: 'calc(1.5em * 3 + 1rem)' }} // Allow vertical resize, set min height based on rows and padding
            // Removed maxHeight and overflow-y-auto as resize handles this
            />

            {/* Send Button */}
            <button
                onClick={handleSubmit}
                className="bg-blue-600 text-white px-4 py-2 rounded self-end disabled:opacity-50 disabled:cursor-not-allowed"
                disabled={!input.trim() && !image}
            >
                Send
            </button>
        </div>
    );
}
