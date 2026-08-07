"use client";

import { useReducer, useRef, useState } from "react";

import MessageList from "@/components/MessageList";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ApiError, streamChat } from "@/lib/api";
import { chatReducer, initialChatState, isTerminalFrame } from "@/lib/chatReducer";
import { STREAM_CUT_OFF, TRANSPORT_FAILURE } from "@/lib/copy";

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

    // Tracked here rather than read back off `state`, which is the snapshot
    // from the render that started this submit and never updates inside the
    // loop. `isTerminalFrame` keeps the rule itself in the reducer.
    let ended = false;

    try {
      for await (const frame of streamChat({ question: asked, sessionId: state.sessionId })) {
        if (isTerminalFrame(frame)) ended = true;
        dispatch({ type: "frame", frame });
        bottom.current?.scrollIntoView({ behavior: "smooth" });
      }

      // A truncated body is not an exception. Under `manage.py runserver` the
      // streaming response carries no Content-Length and no chunked framing, so
      // a body cut short by a generator raising is byte-identical to a complete
      // one at the fetch layer: the reader just finishes and `catch` never runs.
      // Without this the turn stays un-`done`, `streaming` stays true, and the
      // composer is disabled until a full reload throws the transcript away.
      if (!ended) dispatch({ type: "failed", message: STREAM_CUT_OFF });
    } catch (error) {
      dispatch({
        type: "failed",
        // ApiError carries the server's own body, which is written for a user.
        // Anything else is a raw transport rejection ("Failed to fetch") that
        // is not, so it gets the sentence we wrote for that case.
        message: error instanceof ApiError ? error.message : TRANSPORT_FAILURE,
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
