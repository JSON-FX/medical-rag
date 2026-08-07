/**
 * UI-authored PRESENTATION only. Decline sentences come from the server as
 * token text and are rendered as received (spec 5.1) — these are the short
 * labels above that text, never a replacement for it.
 */
export const DECLINE_LABELS: Record<string, string> = {
  empty_corpus: "No documents uploaded",
  off_domain: "Outside your documents",
  weak_unsupported: "Not a confident match",
  insufficient_context: "Not enough detail in your documents",
};

export const FALLBACK_DECLINE_LABEL = "Can't answer from your documents";

export const ERROR_RECOVERY: Record<string, string> = {
  ollama_unavailable:
    "Ollama isn't reachable. Start it, then try again — the answer never left your machine.",
  model_missing:
    "A required model hasn't been pulled yet. The health banner above names the exact command.",
};

export const FALLBACK_ERROR_RECOVERY = "Something went wrong reaching the backend.";

/**
 * The stream ended without a terminal frame — the generator raised mid-answer
 * and the body simply stopped. There is no status code and no error frame to
 * key off, so this is the whole diagnosis the user gets.
 */
export const STREAM_CUT_OFF =
  "The connection dropped before the answer finished. Ask again to retry — nothing above was lost.";

/** fetch itself rejected, so the server never got to say anything usable. */
export const TRANSPORT_FAILURE =
  "Couldn't reach the backend. Check that Django is running, then ask again.";

export function declineLabel(reason: string | null): string {
  if (!reason) return FALLBACK_DECLINE_LABEL;
  return DECLINE_LABELS[reason] ?? FALLBACK_DECLINE_LABEL;
}

export function errorRecovery(code: string): string {
  return ERROR_RECOVERY[code] ?? FALLBACK_ERROR_RECOVERY;
}
