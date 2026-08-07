"use client";

import { useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { uploadDocument } from "@/lib/api";

const MAX_MB = 15;

export default function DocumentUploader({ onUploaded }: { onUploaded: () => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  async function handleChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setError(null);

    // Mirrors the server's rules so the failure is instant rather than a
    // round trip; the server remains the authority (documents/views.py).
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF files are supported.");
      return;
    }
    if (file.size > MAX_MB * 1024 * 1024) {
      setError(`That file is larger than the ${MAX_MB}MB limit.`);
      return;
    }

    setBusy(true);
    try {
      await uploadDocument(file);
      onUploaded();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setBusy(false);
      if (input.current) input.current.value = "";
    }
  }

  return (
    <div className="rounded-lg border p-4">
      <input
        ref={input}
        type="file"
        accept="application/pdf"
        onChange={handleChange}
        disabled={busy}
        className="hidden"
        aria-label="Upload a PDF"
      />
      <Button onClick={() => input.current?.click()} disabled={busy}>
        {busy ? "Processing…" : "Upload a PDF"}
      </Button>
      {busy && (
        // Ingestion is synchronous (PRD 16): extraction, chunking and
        // embedding all happen before this request returns, so a large PDF
        // holds the connection. Saying so keeps it from looking hung.
        <p className="mt-2 text-sm text-muted-foreground">
          Extracting text and generating embeddings. A large PDF can take a minute.
        </p>
      )}
      {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
    </div>
  );
}
