import { useState } from "react";

interface SentenceGrounding {
  sentence: string;
  matched_title: string | null;
  matched_page: number | null;
  similarity: number;
  grounded: boolean;
  citation_mismatch: boolean;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  evidence_status?: string;
  sentence_grounding?: SentenceGrounding[];
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [conversationId, setConversationId] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  async function sendMessage() {
    if (!input.trim()) return;

    const userMessage: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userMessage.content,
          conversation_id: conversationId,
          user_id: 0,
        }),
      });

      const data = await response.json();

      setConversationId(data.conversation_id);

      const assistantMessage: Message = {
        role: "assistant",
        content: data.answer,
        evidence_status: data.evidence_status,
        sentence_grounding: data.sentence_grounding,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error calling MineAI backend:", error);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Error: could not reach MineAI backend." },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") {
      sendMessage();
    }
  }

  return (
    <div style={{ maxWidth: "700px", margin: "40px auto", fontFamily: "sans-serif" }}>
      <h1>MineAI</h1>

      <div style={{ border: "1px solid #ccc", borderRadius: "8px", padding: "16px", minHeight: "300px", marginBottom: "16px" }}>
        {messages.length === 0 && <p style={{ color: "#888" }}>Ask a mining-related question to get started.</p>}

        {messages.map((msg, i) => (
          <div key={i} style={{ marginBottom: "16px" }}>
            <div style={{ fontWeight: "bold", marginBottom: "4px" }}>
              {msg.role === "user" ? "You" : "MineAI"}
            </div>
            <div style={{ whiteSpace: "pre-wrap" }}>{msg.content}</div>

            {msg.evidence_status && (
              <div style={{ marginTop: "6px", fontSize: "0.85em", color: "#555" }}>
                Evidence status: <strong>{msg.evidence_status}</strong>
              </div>
            )}

            {msg.sentence_grounding && msg.sentence_grounding.length > 0 && (
              <details style={{ marginTop: "6px", fontSize: "0.85em" }}>
                <summary style={{ cursor: "pointer", color: "#555" }}>Citations</summary>
                <ul>
                  {msg.sentence_grounding.map((s, j) => (
                    <li key={j} style={{ marginBottom: "4px" }}>
                      {s.matched_title ? `${s.matched_title}, page ${s.matched_page}` : "No source"}
                      {" "}(similarity {s.similarity.toFixed(2)})
                      {s.citation_mismatch && (
                        <span style={{ color: "orange" }}> — citation mismatch flagged</span>
                      )}
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        ))}

        {loading && <div style={{ color: "#888" }}>MineAI is thinking... (this can take a while on CPU)</div>}
      </div>

      <div style={{ display: "flex", gap: "8px" }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask MineAI a question..."
          style={{ flex: 1, padding: "10px", fontSize: "1em" }}
          disabled={loading}
        />
        <button onClick={sendMessage} disabled={loading} style={{ padding: "10px 20px" }}>
          Send
        </button>
      </div>
    </div>
  );
}

export default App;