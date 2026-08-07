"use client";

import { MagnifyingGlass } from "@phosphor-icons/react";
import { useCallback, useEffect, useMemo, useState } from "react";

import DocumentTable from "@/components/DocumentTable";
import DocumentUploader from "@/components/DocumentUploader";
import { Input } from "@/components/ui/input";
import { listDocuments } from "@/lib/api";
import type { DocumentRecord } from "@/lib/types";

type Filter = "all" | DocumentRecord["status"];

const FILTERS: { value: Filter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "ready", label: "Ready" },
  { value: "processing", label: "Processing" },
  { value: "failed", label: "Failed" },
];

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<Filter>("all");

  const refresh = useCallback(() => {
    listDocuments()
      .then((docs) => {
        setDocuments(docs);
        setError(null); // clear a stale failure once things work again
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Could not load documents."));
  }, []);

  useEffect(refresh, [refresh]);

  const shown = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return documents.filter(
      (d) =>
        (filter === "all" || d.status === filter) &&
        (!needle || d.title.toLowerCase().includes(needle)),
    );
  }, [documents, query, filter]);

  const counts = useMemo(
    () => ({
      all: documents.length,
      ready: documents.filter((d) => d.status === "ready").length,
      processing: documents.filter((d) => d.status === "processing").length,
      failed: documents.filter((d) => d.status === "failed").length,
    }),
    [documents],
  );

  return (
    <div className="mx-auto w-full max-w-5xl px-6 py-8">
      <header className="mb-6">
        <h1 className="font-[family-name:var(--font-heading)] text-2xl font-semibold tracking-tight">
          Documents
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Everything the assistant is allowed to answer from. Nothing is sent anywhere —
          text extraction and embedding both run on this machine.
        </p>
      </header>

      <DocumentUploader onUploaded={refresh} />

      {error && (
        <p role="alert" className="mt-4 text-sm text-destructive">
          {error}
        </p>
      )}

      {documents.length > 0 && (
        <div className="mt-6 flex flex-wrap items-center gap-3">
          <div className="relative flex-1 sm:max-w-xs">
            <MagnifyingGlass
              size={16}
              className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground"
              aria-hidden
            />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search by title…"
              aria-label="Search documents by title"
              className="pl-9"
            />
          </div>
          <div className="flex gap-1" role="group" aria-label="Filter by status">
            {FILTERS.map(({ value, label }) => (
              <button
                key={value}
                type="button"
                onClick={() => setFilter(value)}
                aria-pressed={filter === value}
                className={`rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring
                  ${
                    filter === value
                      ? "bg-accent text-accent-foreground"
                      : "text-muted-foreground hover:bg-accent/50"
                  }`}
              >
                {label}
                <span className="ml-1.5 tabular-nums opacity-60">{counts[value]}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="mt-4">
        <DocumentTable documents={shown} onChanged={refresh} />
      </div>
    </div>
  );
}
