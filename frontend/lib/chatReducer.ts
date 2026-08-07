import type { Frame, Source } from "./types";

export type TurnKind = "pending" | "answer" | "decline" | "error";

export interface Turn {
  role: "user" | "assistant";
  text: string;
  kind: TurnKind;
  sources: Source[];
  declineReason: string | null;
  errorCode: string | null;
  /**
   * A recovery sentence written for this specific failure — the server's own
   * error body, or the transport diagnosis the UI made. Null means fall back to
   * the generic copy keyed off `errorCode`.
   */
  errorMessage: string | null;
  truncated: boolean;
  done: boolean;
}

export interface ChatState {
  sessionId: string | null;
  turns: Turn[];
  streaming: boolean;
}

export type ChatAction =
  | { type: "ask"; question: string }
  | { type: "frame"; frame: Frame }
  // The stream did not finish cleanly: either fetch rejected, or the body ended
  // without a terminal frame. Distinct from an `error` frame, which means Django
  // is up and told us what went wrong.
  | { type: "failed"; message: string };

/**
 * Does this frame end the turn?
 *
 * ChatWindow needs this to tell a stream that finished from one that was cut
 * off, and it must agree exactly with which cases below set `done: true` — so
 * the two live next to each other rather than being restated by the caller.
 */
export function isTerminalFrame(frame: Frame): boolean {
  return frame.type === "done" || frame.type === "error";
}

export const initialChatState: ChatState = {
  sessionId: null,
  turns: [],
  streaming: false,
};

function userTurn(text: string): Turn {
  return {
    role: "user",
    text,
    kind: "answer",
    sources: [],
    declineReason: null,
    errorCode: null,
    errorMessage: null,
    truncated: false,
    done: true,
  };
}

function pendingTurn(): Turn {
  return {
    role: "assistant",
    text: "",
    kind: "pending",
    sources: [],
    declineReason: null,
    errorCode: null,
    errorMessage: null,
    truncated: false,
    done: false,
  };
}

/** Replace the last turn via `update`, leaving every other turn untouched. */
function patchLast(state: ChatState, update: (turn: Turn) => Turn): ChatState {
  if (state.turns.length === 0) return state;
  const turns = state.turns.slice();
  turns[turns.length - 1] = update(turns[turns.length - 1]);
  return { ...state, turns };
}

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  if (action.type === "ask") {
    return {
      ...state,
      turns: [...state.turns, userTurn(action.question), pendingTurn()],
      streaming: true,
    };
  }

  if (action.type === "failed") {
    // `text` is deliberately left alone. Whatever streamed before the break is
    // real model output and the user watched it arrive; discarding it would
    // make a cut-off answer indistinguishable from one that never started.
    return {
      ...patchLast(state, (turn) => ({
        ...turn,
        kind: "error",
        errorCode: "transport",
        errorMessage: action.message,
        done: true,
      })),
      streaming: false,
    };
  }

  const frame = action.frame;

  switch (frame.type) {
    case "meta":
      return { ...state, sessionId: frame.session_id };

    case "sources":
      // Sources arrive if and only if the turn will be an answer (spec 3.1).
      return patchLast(state, (turn) => ({
        ...turn,
        kind: "answer",
        sources: frame.items,
      }));

    case "token":
      return patchLast(state, (turn) => ({
        ...turn,
        // Still unclassified when text starts arriving means no sources came
        // first, which by the invariant makes this a decline. Errors are left
        // alone: once a turn has failed, stray tokens do not un-fail it.
        kind: turn.kind === "pending" ? "decline" : turn.kind,
        text: turn.text + frame.text,
      }));

    case "done":
      return {
        ...patchLast(state, (turn) => ({
          ...turn,
          kind: frame.was_declined ? "decline" : turn.kind === "pending" ? "answer" : turn.kind,
          declineReason: frame.decline_reason,
          truncated: frame.truncated,
          done: true,
        })),
        streaming: false,
      };

    case "error":
      // Terminal in its own right. The backend normally sends `done` straight
      // after (chat/views.py), but if the connection drops in between, waiting
      // for it would leave the composer disabled with nothing left to arrive.
      // `errorMessage` stays null: the frame's own message is a diagnostic
      // string, and `code` maps to recovery copy written for a user (copy.ts).
      return {
        ...patchLast(state, (turn) => ({
          ...turn,
          kind: "error",
          errorCode: frame.code,
          done: true,
        })),
        streaming: false,
      };
  }
}
