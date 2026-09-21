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
  missing_information?: string[];
  risk_statements?: string[];
  messageId?: number;
}

function App() {
  const [checkingGapsFor, setCheckingGapsFor] = useState<number | null>(null);
  const [checkingRisksFor, setCheckingRisksFor] = useState<number | null>(null);
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
        messageId: data.message_id,
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

  async function checkGaps(index: number, messageId: number) {
    setCheckingGapsFor(index);
    try {
      const response = await fetch("http://127.0.0.1:8000/check-gaps", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message_id: messageId }),
      });
      const data = await response.json();

      setMessages((prev) =>
        prev.map((msg, i) =>
          i === index ? { ...msg, missing_information: data.missing_information } : msg
        )
      );
    } catch (error) {
      console.error("Error checking gaps:", error);
    } finally {
      setCheckingGapsFor(null);
    }
  }

    async function checkRisks(index: number, messageId: number) {
    setCheckingRisksFor(index);
    try {
      const response = await fetch("http://127.0.0.1:8000/check-risks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message_id: messageId }),
      });
      const data = await response.json();

      setMessages((prev) =>
        prev.map((msg, i) =>
          i === index ? { ...msg, risk_statements: data.risk_statements } : msg
        )
      );
    } catch (error) {
      console.error("Error checking risks:", error);
    } finally {
      setCheckingRisksFor(null);
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

            {msg.role === "assistant" && msg.messageId && !msg.missing_information && (
              <button
                onClick={() => checkGaps(i, msg.messageId!)}
                disabled={checkingGapsFor === i}
                style={{ marginTop: "6px", fontSize: "0.8em", padding: "4px 10px", cursor: "pointer" }}
              >
                {checkingGapsFor === i ? "Checking..." : "Check for gaps"}
              </button>
            )}

            {msg.missing_information && msg.missing_information.length > 0 && (
              <div style={{ marginTop: "6px", fontSize: "0.85em", padding: "8px", background: "#fff8e6", borderRadius: "4px" }}>
                <strong>Gaps in evidence:</strong>
                <ul style={{ margin: "4px 0 0 0", paddingLeft: "18px" }}>
                  {msg.missing_information.map((gap, k) => (
                    <li key={k}>{gap}</li>
                  ))}
                </ul>
              </div>
            )}
             {msg.role === "assistant" && msg.messageId && !msg.risk_statements && (
              <button
                onClick={() => checkRisks(i, msg.messageId!)}
                disabled={checkingRisksFor === i}
                style={{ marginTop: "6px", marginLeft: "8px", fontSize: "0.8em", padding: "4px 10px", cursor: "pointer" }}
              >
                {checkingRisksFor === i ? "Checking..." : "Check for risks"}
              </button>
            )}

            {msg.missing_information && msg.missing_information.length === 0 && (
              <div style={{ marginTop: "6px", fontSize: "0.85em", color: "#888" }}>
                No significant gaps found.
              </div>
            )}
                        {msg.risk_statements && msg.risk_statements.length > 0 && (
              <div style={{ marginTop: "6px", fontSize: "0.85em", padding: "8px", background: "#fdeaea", borderRadius: "4px" }}>
                <strong>Risk-relevant statements in evidence:</strong>
                <ul style={{ margin: "4px 0 0 0", paddingLeft: "18px" }}>
                  {msg.risk_statements.map((risk, k) => (
                    <li key={k}>{risk}</li>
                  ))}
                </ul>
              </div>   
            )}

            {msg.risk_statements && msg.risk_statements.length === 0 && (
              <div style={{ marginTop: "6px", fontSize: "0.85em", color: "#888" }}>
                No risk-relevant statements found.
              </div>
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