"use client";

import { PaperPlaneTilt } from "@phosphor-icons/react";
import { useEffect, useReducer, useRef, useState } from "react";

import EvidencePanel from "@/components/EvidencePanel";
import MessageBubble from "@/components/MessageBubble";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { streamChat } from "@/lib/api";
import { chatReducer, initialChatState, isTerminalFrame } from "@/lib/chatReducer";

const STREAM_CUT_OFF =
  "The connection closed before the answer finished. Ask again to retry.";

const EXAMPLES = [
  "What is the adult starting dose of metformin?",
  "When is celecoxib contraindicated?",
  "What is the capital of France?",
];

export default function ChatWindow() {
  const [state, dispatch] = useReducer(chatReducer, initialChatState);
  const [question, setQuestion] = useState("");
  // Which assistant turn the evidence panel is showing, and which of its
  // sources is highlighted. Selection lives here rather than in the reducer:
  // it is view state, and the reducer stays pure and testable without it.
  const [selectedTurn, setSelectedTurn] = useState<number | null>(null);
  const [activeSource, setActiveSource] = useState<number | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  const turns = state.turns;
  const lastAssistant = turns.map((t, i) => ({ t, i })).filter(({ t }) => t.role === "assistant").pop();
  const shownIndex = selectedTurn ?? lastAssistant?.i ?? null;
  const shownTurn = shownIndex === null ? null : (turns[shownIndex] ?? null);

  // Follow the newest answer unless the reader has deliberately selected an
  // older one.
  useEffect(() => {
    if (selectedTurn === null) setActiveSource(null);
  }, [turns.length, selectedTurn]);

  async function ask(asked: string) {
    if (!asked || state.streaming) return;
    setQuestion("");
    setSelectedTurn(null);
    setActiveSource(null);
    dispatch({ type: "ask", question: asked });

    let ended = false;
    try {
      for await (const frame of streamChat({ question: asked, sessionId: state.sessionId })) {
        if (isTerminalFrame(frame)) ended = true;
        dispatch({ type: "frame", frame });
        bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
      }
      // A truncated body is byte-identical to a clean end of stream at the
      // fetch layer, so a stream that dies without a terminal frame would
      // otherwise leave the composer disabled forever.
      if (!ended) dispatch({ type: "failed", message: STREAM_CUT_OFF });
    } catch (error) {
      dispatch({
        type: "failed",
        message: error instanceof Error ? error.message : "Request failed",
      });
    }
  }

  return (
    <div className="grid h-full grid-cols-1 xl:grid-cols-[minmax(0,1fr)_380px]">
      <section className="flex min-h-0 flex-col">
        <div className="min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-3xl px-6 py-8">
            {turns.length === 0 ? (
              <div className="py-10">
                <h1 className="font-[family-name:var(--font-heading)] text-2xl font-semibold tracking-tight">
                  Ask your documents
                </h1>
                <p className="mt-2 max-w-prose text-sm text-muted-foreground">
                  Answers are drawn only from the PDFs you upload, and cite the page they
                  came from. When your documents don&apos;t support an answer, you get a
                  decline instead of a guess.
                </p>
                <ul className="mt-6 flex flex-col gap-2">
                  {EXAMPLES.map((example) => (
                    <li key={example}>
                      <button
                        type="button"
                        onClick={() => ask(example)}
                        className="w-full rounded-lg border px-3.5 py-2.5 text-left text-sm transition-colors hover:border-primary/40 hover:bg-accent/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                      >
                        {example}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <div className="flex flex-col gap-5">
                {turns.map((turn, i) => (
                  <MessageBubble
                    key={i}
                    turn={turn}
                    selected={i === shownIndex}
                    activeIndex={activeSource}
                    onCite={(index) => {
                      setSelectedTurn(i);
                      setActiveSource(index);
                    }}
                    onSelect={() => setSelectedTurn(i)}
                  />
                ))}
              </div>
            )}
            <div ref={bottom} />
          </div>
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            ask(question.trim());
          }}
          className="border-t bg-background/80 px-6 py-4 backdrop-blur"
        >
          <div className="mx-auto flex max-w-3xl gap-2">
            <Input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask about your uploaded documents…"
              disabled={state.streaming}
              aria-label="Question"
              className="h-11"
            />
            <Button
              type="submit"
              disabled={state.streaming || !question.trim()}
              className="h-11 px-4"
            >
              <PaperPlaneTilt size={16} weight="fill" aria-hidden />
              <span className="ml-2">{state.streaming ? "Answering…" : "Ask"}</span>
            </Button>
          </div>
        </form>
      </section>

      {/* Below xl the panel stacks under the transcript rather than vanishing:
          the evidence is the product, not a desktop luxury. */}
      <div className="hidden min-h-0 xl:block">
        <EvidencePanel
          turn={shownTurn}
          activeIndex={activeSource}
          onSelect={setActiveSource}
        />
      </div>
    </div>
  );
}
