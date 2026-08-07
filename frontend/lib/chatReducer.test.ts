import { describe, expect, test } from "vitest";

import { chatReducer, initialChatState, type ChatAction, type ChatState } from "./chatReducer";
import type { Frame, Source } from "./types";

const SOURCE: Source = {
  chunk_id: "1_0",
  document_id: 1,
  title: "Metformin.pdf",
  page: 3,
  snippet: "Starting dose 500 mg",
};

function run(actions: ChatAction[]): ChatState {
  return actions.reduce(chatReducer, initialChatState);
}

const ask = (question: string): ChatAction => ({ type: "ask", question });
const frame = (f: Frame): ChatAction => ({ type: "frame", frame: f });

describe("turn bookkeeping", () => {
  test("asking appends a user turn and a pending assistant turn", () => {
    const state = run([ask("What is the dose?")]);
    expect(state.turns).toHaveLength(2);
    expect(state.turns[0]).toMatchObject({ role: "user", text: "What is the dose?" });
    expect(state.turns[1]).toMatchObject({ role: "assistant", kind: "pending", done: false });
    expect(state.streaming).toBe(true);
  });

  test("meta records the session id for later turns", () => {
    const state = run([ask("q"), frame({ type: "meta", session_id: "s-1" })]);
    expect(state.sessionId).toBe("s-1");
  });
});

describe("answers", () => {
  test("a sources frame marks the turn an answer and attaches citations", () => {
    const state = run([
      ask("q"),
      frame({ type: "meta", session_id: "s-1" }),
      frame({ type: "sources", items: [SOURCE] }),
      frame({ type: "token", text: "500 mg" }),
    ]);
    const turn = state.turns[1];
    expect(turn.kind).toBe("answer");
    expect(turn.sources).toEqual([SOURCE]);
    expect(turn.text).toBe("500 mg");
  });

  test("tokens accumulate in order", () => {
    const state = run([
      ask("q"),
      frame({ type: "sources", items: [SOURCE] }),
      frame({ type: "token", text: "500 " }),
      frame({ type: "token", text: "mg" }),
    ]);
    expect(state.turns[1].text).toBe("500 mg");
  });

  test("done ends the turn and stops streaming", () => {
    const state = run([
      ask("q"),
      frame({ type: "sources", items: [SOURCE] }),
      frame({ type: "token", text: "500 mg" }),
      frame({
        type: "done",
        message_id: 7,
        was_declined: false,
        decline_reason: null,
        truncated: false,
      }),
    ]);
    expect(state.turns[1]).toMatchObject({ kind: "answer", done: true, truncated: false });
    expect(state.streaming).toBe(false);
  });

  test("a truncated done marks the answer incomplete", () => {
    const state = run([
      ask("q"),
      frame({ type: "sources", items: [SOURCE] }),
      frame({ type: "token", text: "partial" }),
      frame({
        type: "done",
        message_id: 8,
        was_declined: false,
        decline_reason: null,
        truncated: true,
      }),
    ]);
    expect(state.turns[1].truncated).toBe(true);
  });
});

describe("declines", () => {
  test("a token with no preceding sources is a decline", () => {
    // The invariant from spec 3.1: sources arrive if and only if the turn is
    // an answer. This is what lets a decline render as a decline from its
    // first character instead of restyling when done arrives.
    const state = run([
      ask("capital of France?"),
      frame({ type: "meta", session_id: "s-1" }),
      frame({ type: "token", text: "I can only answer questions grounded in..." }),
    ]);
    expect(state.turns[1].kind).toBe("decline");
  });

  test("done supplies the decline reason", () => {
    const state = run([
      ask("q"),
      frame({ type: "token", text: "copy" }),
      frame({
        type: "done",
        message_id: 9,
        was_declined: true,
        decline_reason: "off_domain",
        truncated: false,
      }),
    ]);
    expect(state.turns[1]).toMatchObject({
      kind: "decline",
      declineReason: "off_domain",
      done: true,
    });
  });

  test("a decline that somehow arrives with no tokens is still a decline", () => {
    const state = run([
      ask("q"),
      frame({
        type: "done",
        message_id: 10,
        was_declined: true,
        decline_reason: "empty_corpus",
        truncated: false,
      }),
    ]);
    expect(state.turns[1].kind).toBe("decline");
  });

  test("server decline copy is preserved verbatim", () => {
    const copy = "No documents have been uploaded yet. Upload a medical reference document";
    const state = run([ask("q"), frame({ type: "token", text: copy })]);
    expect(state.turns[1].text).toBe(copy);
  });
});

describe("errors", () => {
  test("an error frame marks the turn an error and records the code", () => {
    const state = run([
      ask("q"),
      frame({ type: "error", code: "ollama_unavailable", message: "connection refused" }),
    ]);
    expect(state.turns[1]).toMatchObject({ kind: "error", errorCode: "ollama_unavailable" });
  });

  test("a transport failure with no frames at all is an error turn", () => {
    const state = run([ask("q"), { type: "failed", message: "Failed to fetch" }]);
    expect(state.turns[1]).toMatchObject({ kind: "error", done: true });
    expect(state.streaming).toBe(false);
  });

  test("an error frame does not get reclassified as a decline by later tokens", () => {
    const state = run([
      ask("q"),
      frame({ type: "error", code: "model_missing", message: "not found" }),
      frame({ type: "token", text: "stray" }),
    ]);
    expect(state.turns[1].kind).toBe("error");
  });

  test("the done frame that follows an error does not turn it into an answer", () => {
    // The backend's error path emits error then done{was_declined:false}
    // (chat/views.py), so this exact sequence happens whenever Ollama is
    // down. A turn that flipped to "answer" here would render an empty
    // answer bubble instead of the recovery message.
    const state = run([
      ask("q"),
      frame({ type: "error", code: "ollama_unavailable", message: "refused" }),
      frame({
        type: "done",
        message_id: 11,
        was_declined: false,
        decline_reason: null,
        truncated: true,
      }),
    ]);
    expect(state.turns[1]).toMatchObject({ kind: "error", done: true });
    expect(state.streaming).toBe(false);
  });
});

describe("purity", () => {
  test("the reducer does not mutate the state it is given", () => {
    const before = run([ask("q")]);
    const snapshot = JSON.stringify(before);
    chatReducer(before, frame({ type: "token", text: "x" }));
    expect(JSON.stringify(before)).toBe(snapshot);
  });
});
