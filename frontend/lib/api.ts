import { readFrames } from "./ndjson";
import type { DocumentRecord, Frame, HealthReport } from "./types";

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

/** Error carrying the server's own message, which is written for a user. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function failure(response: Response): Promise<ApiError> {
  let message = `Request failed (${response.status})`;
  try {
    const body = await response.json();
    if (typeof body?.error === "string") message = body.error;
  } catch {
    // Non-JSON error body: keep the status-based message.
  }
  return new ApiError(message, response.status);
}

export async function getHealth(): Promise<HealthReport> {
  const response = await fetch(`${API_BASE}/api/health/`, { cache: "no-store" });
  if (!response.ok) throw await failure(response);
  return response.json();
}

export async function listDocuments(): Promise<DocumentRecord[]> {
  const response = await fetch(`${API_BASE}/api/documents/`, { cache: "no-store" });
  if (!response.ok) throw await failure(response);
  return response.json();
}

export async function uploadDocument(file: File): Promise<DocumentRecord> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_BASE}/api/documents/`, { method: "POST", body: form });
  if (!response.ok) throw await failure(response);
  return response.json();
}

export async function deleteDocument(id: number): Promise<void> {
  const response = await fetch(`${API_BASE}/api/documents/${id}/`, { method: "DELETE" });
  if (!response.ok) throw await failure(response);
}

/**
 * POST a question and yield NDJSON frames as they arrive.
 *
 * No proxy layer: this reads Django's stream directly, so the first token is
 * not gated on the whole response arriving (spec 2.1).
 */
export async function* streamChat(options: {
  question: string;
  sessionId: string | null;
  signal?: AbortSignal;
}): AsyncGenerator<Frame> {
  const response = await fetch(`${API_BASE}/api/chat/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: options.question,
      ...(options.sessionId ? { session_id: options.sessionId } : {}),
    }),
    signal: options.signal,
  });

  if (!response.ok) throw await failure(response);
  if (!response.body) throw new ApiError("Response had no body to stream", response.status);

  yield* readFrames(response.body);
}
