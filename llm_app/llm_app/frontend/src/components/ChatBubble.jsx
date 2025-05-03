import React, { useRef, useState } from "react"; // Import useRef
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";

// Define a custom component for rendering code blocks
const CodeBlock = ({ node, inline, className, children, ...props }) => {
    // Extract language from className if it exists (e.g., "language-js")
    const match = /language-(\w+)/.exec(className || '');
    const isBlock = !inline && match; // Check if it's a block code element
    const language = isBlock ? match[1] : null; // Get the language identifier

    // State for copy button visibility and text (e.g., "Copy" or "Copied!")
    const [showControls, setShowControls] = useState(false);
    const [copyStatus, setCopyStatus] = useState('Copy');

    // Create a ref to hold the DOM element of the code block
    const codeElementRef = useRef(null);

    const handleCopy = async () => {
        // Get the text content directly from the rendered DOM element using the ref
        const codeTextToCopy = codeElementRef.current ? codeElementRef.current.textContent : '';

        // Check if text is empty BEFORE attempting to copy
        if (!codeTextToCopy) {
            console.warn("Cannot copy: Rendered code block text is empty.");
            // No need for alert, just log and return
            return;
        }

        console.log("Attempting to copy rendered text:", codeTextToCopy); // Log the text being copied

        try {
            // Use the standard Clipboard API to write text
            await navigator.clipboard.writeText(codeTextToCopy);
            console.log("Successfully copied!"); // Log success
            setCopyStatus('Copied!');
            // Reset status after a brief period
            setTimeout(() => setCopyStatus('Copy'), 2000);
        } catch (err) {
            console.error('Failed to copy code: ', err); // Log the actual error object
            setCopyStatus('Failed!'); // Status for Clipboard API error
            // Reset status even on failure
            setTimeout(() => setCopyStatus('Copy'), 2000);

            // Provide more specific user feedback based on the error type or environment
            if (err.name === 'NotAllowedError') {
                alert("Copying to clipboard was denied by the browser. This often happens if the page is not focused or lacks necessary permissions.");
            } else if (navigator.clipboard === undefined) {
                alert("Clipboard API not available in this browser or context. Please update your browser.");
            } else if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost') {
                alert("Copying to clipboard requires a secure connection (HTTPS) or localhost. You are currently using: " + window.location.protocol);
            } else {
                // Generic failure message for other errors
                alert("Failed to copy to clipboard. Please check the browser console for error details.");
            }
        }
    };

    if (isBlock) {
        // Render a block code component with language label and copy button
        return (
            // Container for positioning controls relative to the code block area
            <div
                className="relative my-2 rounded bg-gray-800 dark:bg-gray-900"
                onMouseEnter={() => setShowControls(true)} // Show controls on hover
                onMouseLeave={() => setShowControls(false)} // Hide controls on mouse leave
            >
                {/* Render the pre and code tags with highlighting */}
                {/* NOTE: <pre> inherently preserves whitespace and newlines */}
                <pre className="overflow-x-auto p-4">
                    {/* Attach the ref to the code element */}
                    <code ref={codeElementRef} className={className} {...props}>
                        {children} {/* Render the highlighted code */}
                    </code>
                </pre>

                {/* Container for Language Label and Copy Button */}
                {/* Show controls on hover OR if copy status is not 'Copy' */}
                {(showControls || copyStatus !== 'Copy') && (
                    <div className="absolute top-2 right-2 flex items-center gap-2">
                        {/* Language Label - Only display if language is found */}
                        {language && (
                            <span className="text-white text-xs px-2 py-1 rounded bg-gray-700 opacity-80">
                                {language}
                            </span>
                        )}
                        {/* Copy button */}
                        <button
                            className="bg-gray-700 hover:bg-gray-600 text-white text-xs px-2 py-1 rounded opacity-80 hover:opacity-100 transition-opacity"
                            onClick={handleCopy}
                            title="Copy to clipboard"
                        >
                            {copyStatus}
                        </button>
                    </div>
                )}
            </div>
        );
    } else {
        // Render inline code (no controls needed)
        return (
            <code className="bg-gray-300 dark:bg-gray-600 px-1 rounded text-sm" {...props}>
                {children}
            </code>
        );
    }
};


export default function ChatBubble({ role, content, image }) {
    const isUser = role === "user";
    const bubbleClasses = isUser
        ? "bg-blue-600 text-white ml-auto"
        : "bg-gray-700 text-white mr-auto";

    return (
        <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
            <div className={`rounded-lg p-3 break-words ${bubbleClasses} max-w-xs sm:max-w-sm md:max-w-md lg:max-w-lg`}>

                {image && (
                    <img
                        src={URL.createObjectURL(image)}
                        alt="Uploaded content"
                        className="max-w-full h-auto max-h-64 rounded-md mb-2"
                    />
                )}
                {/* Add whitespace-pre-line here */}
                <div className="whitespace-pre-line">
                    <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={[rehypeHighlight]}
                        components={{
                            code: CodeBlock, // Use our custom CodeBlock component
                            // Pass through pre, ReactMarkdown handles it internally with CodeBlock
                            pre: ({ node, children, ...props }) => <>{children}</>, // Render children directly, CodeBlock wraps in <pre>
                            // Keep other custom table components
                            table: ({ node, ...props }) => (
                                <table className="table-auto w-full border-collapse my-2 bg-white dark:bg-gray-800 text-black dark:text-white" {...props} />
                            ),
                            thead: ({ node, ...props }) => (
                                <thead className="bg-gray-200 dark:bg-gray-600" {...props} />
                            ),
                            tbody: ({ node, ...props }) => (
                                <tbody {...props} />
                            ),
                            tr: ({ node, ...props }) => (
                                <tr className="even:bg-gray-100 dark:even:bg-gray-700" {...props} />
                            ),
                            th: ({ node, ...props }) => (
                                <th className="px-4 py-2 text-left border-b-2 border-gray-400 dark:border-gray-500 font-bold" {...props} />
                            ),
                            td: ({ node, ...props }) => (
                                <td className="px-4 py-2 border-b border-gray-300 dark:border-gray-600" {...props} />
                            ),
                        }}
                    >
                        {content}
                    </ReactMarkdown>
                </div>
            </div>
        </div>
    );
}
