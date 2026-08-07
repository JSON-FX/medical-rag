"use client";

import { CheckCircle, Trash, WarningOctagon } from "@phosphor-icons/react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { deleteDocument } from "@/lib/api";
import type { DocumentRecord } from "@/lib/types";

const STATUS = {
  ready: { icon: CheckCircle, className: "text-emerald-600 dark:text-emerald-400", label: "Ready" },
  processing: { icon: CheckCircle, className: "text-amber-600 dark:text-amber-400", label: "Processing" },
  failed: { icon: WarningOctagon, className: "text-destructive", label: "Failed" },
} as const;

export default function DocumentTable({
  documents,
  onChanged,
}: {
  documents: DocumentRecord[];
  onChanged: () => void;
}) {
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<number | null>(null);

  async function remove(id: number) {
    // Deleting is destructive and irreversible, so a failure that looks
    // identical to success is the worst outcome. Surface it, and only refresh
    // when something actually happened.
    setError(null);
    setBusyId(id);
    try {
      await deleteDocument(id);
      onChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed.");
    } finally {
      setBusyId(null);
    }
  }

  if (documents.length === 0) {
    return (
      <div className="rounded-xl border border-dashed px-6 py-14 text-center">
        <p className="text-sm font-medium">No documents match</p>
        <p className="mt-1 text-sm text-muted-foreground">
          Upload a PDF, or clear the filters above.
        </p>
      </div>
    );
  }

  return (
    <>
      {error && (
        <p role="alert" className="mb-3 text-sm text-destructive">
          {error}
        </p>
      )}
      <div className="overflow-hidden rounded-xl border">
        <table className="w-full text-sm">
          <caption className="sr-only">Uploaded documents</caption>
          <thead>
            <tr className="border-b bg-muted/50 text-left text-xs text-muted-foreground">
              <th scope="col" className="px-4 py-2.5 font-medium">Title</th>
              <th scope="col" className="px-4 py-2.5 font-medium">Status</th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">Pages</th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">Chunks</th>
              <th scope="col" className="px-4 py-2.5"><span className="sr-only">Actions</span></th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {documents.map((doc) => {
              const status = STATUS[doc.status];
              const Icon = status.icon;
              return (
                <tr key={doc.id} className="transition-colors hover:bg-accent/40">
                  <td className="px-4 py-3">
                    <span className="font-medium">{doc.title}</span>
                    {doc.status === "failed" && doc.error_message && (
                      <p className="mt-0.5 text-xs text-destructive">{doc.error_message}</p>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    {/* Icon plus text, never colour alone. */}
                    <span className={`inline-flex items-center gap-1.5 text-xs font-medium ${status.className}`}>
                      <Icon size={14} weight="fill" aria-hidden />
                      {status.label}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums text-muted-foreground">
                    {doc.page_count ?? "—"}
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums text-muted-foreground">
                    {doc.chunk_count}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <Button
                      variant="ghost"
                      size="sm"
                      disabled={busyId === doc.id}
                      onClick={() => remove(doc.id)}
                      aria-label={`Delete ${doc.title}`}
                      className="text-muted-foreground hover:text-destructive"
                    >
                      <Trash size={15} aria-hidden />
                      <span className="ml-1.5">{busyId === doc.id ? "Deleting…" : "Delete"}</span>
                    </Button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}
