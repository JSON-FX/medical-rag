export interface Source {
  chunk_id: string;
  document_id: number;
  title: string;
  page: number;
  snippet: string;
}

export type Frame =
  | { type: "meta"; session_id: string }
  | { type: "sources"; items: Source[] }
  | { type: "token"; text: string }
  | {
      type: "done";
      message_id: number;
      was_declined: boolean;
      decline_reason: string | null;
      truncated: boolean;
    }
  | { type: "error"; code: string; message: string };

export interface HealthReport {
  ollama_reachable: boolean;
  host: string;
  models: { chat: boolean; embed: boolean };
  expected: { chat: string; embed: string };
  documents_ready: number;
}

export interface DocumentRecord {
  id: number;
  title: string;
  status: "processing" | "ready" | "failed";
  page_count: number | null;
  chunk_count: number;
  uploaded_at: string;
  error_message: string;
}
