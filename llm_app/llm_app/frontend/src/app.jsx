import { useEffect, useState } from "react";

import ChatBox from "./components/ChatBox"; // Assuming ChatBox is correct
import InputBar from "./components/InputBar"; // Assuming InputBar is correct

const STATIC_MODELS = {
    "local_huggingface": [
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "llava-hf/llava-1.5-7b-hf",           // faster but not good with RAG or chat
        // "Qwen/Qwen2-VL-7B-Instruct",         // not working very well with img
    ],
    "openrouter_api": [
        "mistralai/mistral-7b-instruct",
        "deepseek/deepseek-chat-v3-0324",
        "meta-llama/llama-3-8b-instruct",
        "meta-llama/llama-4-scout",
    ],
};


export default function App() {
    // loading screen
    const [loading, setLoading] = useState(true);

    // handle conversations and history
    const [conversationName, setConversationName] = useState("");
    const [availableConversations, setAvailableConversations] = useState([]);

    // control llm model used
    const [backend, setMode] = useState("local_huggingface");
    const [selectedModel, setSelectedModel] = useState(STATIC_MODELS[backend]?.[0] || null);

    // control llm generation
    const [systemPrompt, setSystemPrompt] = useState("");
    const [messages, setMessages] = useState([]);
    const [image, setImage] = useState(null);
    const [temperature, setTemperature] = useState(0.25);
    const [maxTokens, setMaxTokens] = useState(500);

    const refreshConversations = async () => {
        try {
            const res = await fetch("http://localhost:8000/chat_list_conversations");
            const data = await res.json();
            setAvailableConversations(data);
            await setChatHistory();
        } catch (err) {
            console.info("Failed to refresh conversations:", err);
        }
    };

    const setChatHistory = async () => {
        const responses = await queryBackend("[HISTORY]");
        const listMessages = responses["response"]

        setSystemPrompt("");
        setMessages([]);
        setImage(null);

        for (const parsedMsg of listMessages) {

            if (!parsedMsg || !parsedMsg.role || !parsedMsg.content) {
                console.warn("Skipped bad message:", raw_msg);
                continue;
            }

            try {
                const role = parsedMsg.role;
                const contentArr = parsedMsg.content;
                const text_content = contentArr.find(c => c.type === "text")?.text || "";
                const hasImage = contentArr.some(c => c.type === "image");

                let out_msg = { role, content: text_content };
                if (hasImage) {
                    out_msg = { role, content: "[🖼️] " + text_content };
                }

                if (role === "system") {
                    setSystemPrompt(text_content);
                } else {
                    setMessages(prev => [...prev, out_msg]);
                }

            } catch (err) {
                console.error("Error rendering message:", err);
            }
        }
    };


    const handleReset = () => {
        setSystemPrompt("");
        setMessages([]);
        setImage(null);
        handleSend("[RESET]");
        refreshConversations();
        setConversationName("")
    };

    const handleHistory = () => {
        handleSend("[HISTORY]");
    };

    const queryBackend = async (text, options = {}) => {
        const {
            overrideConversationName = conversationName,
            overrideSystemPrompt = systemPrompt,
            overrideModelId = selectedModel,
        } = options;

        const formData = new FormData();
        formData.append("conversation_name", overrideConversationName !== "anonymous" ? overrideConversationName : "");
        formData.append("system_message", overrideSystemPrompt);
        formData.append("model_id", overrideModelId);
        formData.append("message", text);
        formData.append("temperature", temperature);
        formData.append("max_tokens", maxTokens);
        if (image) formData.append("image_file", image);

        const endpoint =
            backend === "local_huggingface"
                ? "http://localhost:8000/chat_local_huggingface"
                : "http://localhost:8000/chat_openrouter_api";

        try {
            const res = await fetch(endpoint, {
                method: "POST",
                body: formData,
            });

            if (!res.ok) { throw new Error(`HTTP error! status: ${res.status}`); }
            const data = await res.json();
            return data;

        } catch (error) {
            console.info("Error in queryBackend:", error);
        }
    }

    const handleSend = async (text, options = {}) => {

        try {
            if (text !== "[RESET]" && text !== "[HISTORY]" && text !== "[CONVERSATION]") {
                const newMsg = { role: "user", content: text, image };
                setMessages((prev) => [...prev, newMsg]);
            }
        } catch (error) {
            console.info("ERROR setting message", error.message);
        }

        try {
            const data = await queryBackend(text, options);

            setMessages((prev) => [
                ...prev,
                { role: "assistant", content: data.response },
            ]);

        } catch (error) {
            console.info("ERROR sending message: ", error.message);
            setMessages((prev) => [
                ...prev,
                {
                    role: "assistant",
                    content: `[Error contacting backend: ${error.message}]`,
                },
            ]);
        } finally {
            setImage(null);
        }
    };


    useEffect(() => {
        const initApp = async () => {
            try {
                await refreshConversations();
            } catch (e) {
                console.error("App init failed:", e);
            } finally {
                setLoading(false);
            }
        };
        initApp();
    }, []);

    // render loading screen while waiting for backend to load
    if (loading) {
        return (
            <div className="flex flex-col justify-center items-center h-screen bg-black text-white text-xl">
                <div className="flex gap-10">
                    <div className="animate-spin border-4 border-white border-t-transparent rounded-full h-20 w-20"></div>
                    <span>
                        🧠 LLM Chat App is loading...
                        <br></br>
                        <i>
                            <br></br> ... 🧙🏻‍♂️⏰ please wait while the backend wizards do their magic 🧙🏾‍♂️🦄
                            <br></br> ... perhaps try to refresh the page from time to time to let them know you're waiting ...
                            <br></br>
                            <br></br> ... if its your first time running this thing, it might take a while to download all required artefacts,
                            <br></br> especially if your internet packets come by carrier pigeon...
                            <br></br>
                            <br></br> ... meanwhile ponder about your life choices with this LLM-generated haiku about the app:

                            <br></br>
                            <br></br>❝❝❝ Screen holds frozen breath,
                            <br></br>Code's deep flaws, a slow decay,
                            <br></br>Patience wears so thin.
                            <br></br>No worth found in this long wait,
                            <br></br>Just errors and wasted time.
                            <br></br>
                            <br></br>Hours slowly crawl,
                            <br></br>Hopes of function, now all gone,
                            <br></br>Just a hollow shell.
                            <br></br>Frustration builds, a rising tide,
                            <br></br>This poor app, a broken dream.
                            <br></br>
                            <br></br>Cursor blinks and waits,
                            <br></br>No clever thought, no swift reply,
                            <br></br>Just a silent void.
                            <br></br>A digital ghost it seems,
                            <br></br>Forever stuck, forever slow. ❞❞❞
                            <br></br>- Gemini Flash 2.5
                        </i>
                    </span>
                </div>
            </div>
        );
    }

    // render default interface
    return (
        <div className="flex flex-col h-screen bg-gray-100 dark:bg-gray-900 text-black dark:text-white">
            <header className="p-4 border-b border-gray-700 flex justify-between items-start flex-wrap gap-4 text-sm">

                <div className="flex flex-col gap-1">
                    {/* App title */}
                    <div className="flex items-center gap-2">
                        <img src="/BrainLlmAPP.png" alt="Logo" className="h-15 w-20" />
                        <h1 className="text-xl font-bold">LLM Chat App
                            <p className="text-xs text-gray-600 italic">
                                The best shitty ChatGPT clone you'll find today 💩✨
                            </p>
                        </h1>
                    </div>

                    {/* Conversation Selector */}
                    <label className="text-xs text-gray-400">Conversation</label>
                    <select
                        value={conversationName}
                        onChange={async (e) => {
                            const val = e.target.value;
                            if (val === "__new__") {
                                const now = new Date().toISOString().split('T')[0];
                                const userInput = prompt("Enter new conversation name:");
                                const name = userInput ? `(${now}): ${userInput}` : null;
                                if (name) {
                                    setConversationName(name);
                                    setMessages([]);
                                    handleSend("[CONVERSATION]", { overrideConversationName: name });
                                    await refreshConversations();
                                }
                            } else {
                                setConversationName(val);
                                setMessages([]);
                                handleSend("[CONVERSATION]", { overrideConversationName: val });
                                await refreshConversations();
                            }
                        }}
                        className="bg-gray-800 text-white rounded px-2 py-1"
                    >
                        <option value="anonymous">👻 Anonymous</option>
                        <option value="__new__">➕ New Conversation</option>
                        {availableConversations.map((conv) => (
                            <option key={conv} value={conv}>💬 {conv}</option>
                        ))}
                    </select>
                </div>

                {/* Backend + Model Group */}
                <div className="flex flex-col gap-1">
                    <label className="text-xs text-gray-400">Backend</label>
                    <select
                        value={backend}
                        onChange={(e) => {
                            const newMode = e.target.value;
                            setMode(newMode);
                            setSelectedModel(STATIC_MODELS[newMode]?.[0] || null);
                        }}
                        className="bg-gray-800 text-white rounded px-2 py-1"
                    >
                        <option value="local_huggingface">LocalHuggingface</option>
                        <option value="openrouter_api">OpenRouterAPI</option>
                    </select>

                    <label className="text-xs text-gray-400">Model</label>
                    <select
                        value={selectedModel || ''}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        className="bg-gray-800 text-white rounded px-2 py-1"
                        disabled={!STATIC_MODELS[backend]?.length}
                    >
                        {!STATIC_MODELS[backend]?.length ? (
                            <option value="">No models available</option>
                        ) : (
                            STATIC_MODELS[backend].map(model => (
                                <option key={model} value={model}>{model}</option>
                            ))
                        )}
                    </select>
                </div>

                {/* Generation Controls */}
                <div className="flex flex-col gap-1">
                    <label className="text-xs text-gray-400">Temperature</label>
                    <input
                        type="number"
                        value={temperature}
                        onChange={(e) => setTemperature(parseFloat(e.target.value))}
                        min="0"
                        max="1.5"
                        step="0.05"
                        className="w-20 px-2 py-0 rounded border bg-white text-black"
                    />

                    <label className="text-xs text-gray-400">Max Tokens</label>
                    <input
                        type="number"
                        value={maxTokens}
                        onChange={(e) => setMaxTokens(Number(e.target.value))}
                        min="50"
                        max="1000"
                        step="25"
                        className="w-20 px-2 py-0 rounded border bg-white text-black"
                    />
                </div>

                {/* Actions */}
                <div className="flex flex-col gap-2 py-4">
                    <button
                        onClick={handleReset}
                        className="bg-red-600 hover:bg-red-700 text-white px-3 py-1 rounded"
                    >
                        Reset
                    </button>
                </div>

            </header >

            {/* System Prompt UI */}
            < div className="flex items-center gap-2 p-2 bg-gray-200 dark:bg-gray-800 border-b border-gray-700" >
                <label className="text-xs font-semibold text-gray-700 dark:text-gray-300">
                    System Prompt:
                </label>
                <textarea
                    value={systemPrompt}
                    onChange={(e) => setSystemPrompt(e.target.value)}
                    rows={1}
                    className="flex-1 text-xs bg-white dark:bg-gray-900 dark:text-white border border-gray-400 dark:border-gray-600 rounded px-2 py-0.5 resize-none"
                    placeholder="e.g. Respond like a pirate who enjoys sea shanties a bit too much..."
                />
            </div >
            <ChatBox messages={messages} />
            <InputBar onSend={handleSend} image={image} setImage={setImage} conversationName={conversationName} />
        </div >
    );
}
