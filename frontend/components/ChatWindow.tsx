"use client";

import { useReducer, useRef, useState } from "react";

import MessageList from "@/components/MessageList";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { streamChat } from "@/lib/api";
import { chatReducer, initialChatState } from "@/lib/chatReducer";

export default function ChatWindow() {
  const [state, dispatch] = useReducer(chatReducer, initialChatState);
  const [question, setQuestion] = useState("");
  const bottom = useRef<HTMLDivElement>(null);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    const asked = question.trim();
    if (!asked || state.streaming) return;

    setQuestion("");
    dispatch({ type: "ask", question: asked });

    try {
      for await (const frame of streamChat({ question: asked, sessionId: state.sessionId })) {
        dispatch({ type: "frame", frame });
        bottom.current?.scrollIntoView({ behavior: "smooth" });
      }
    } catch (error) {
      dispatch({
        type: "failed",
        message: error instanceof Error ? error.message : "Request failed",
      });
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <MessageList turns={state.turns} />
      <div ref={bottom} />
      <form onSubmit={submit} className="sticky bottom-4 flex gap-2 bg-background pt-2">
        <Input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about your uploaded documents…"
          disabled={state.streaming}
          aria-label="Question"
        />
        <Button type="submit" disabled={state.streaming || !question.trim()}>
          {state.streaming ? "Answering…" : "Ask"}
        </Button>
      </form>
    </div>
  );
}
