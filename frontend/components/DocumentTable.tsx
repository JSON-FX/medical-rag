"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";
import { deleteDocument } from "@/lib/api";
import type { DocumentRecord } from "@/lib/types";

const STATUS_STYLES: Record<DocumentRecord["status"], string> = {
  ready: "text-green-700 dark:text-green-400",
  processing: "text-amber-700 dark:text-amber-400",
  failed: "text-red-700 dark:text-red-400",
};

export default function DocumentTable({
  documents,
  onChanged,
}: {
  documents: DocumentRecord[];
  onChanged: () => void;
}) {
  const [error, setError] = useState<string | null>(null);

  async function remove(id: number) {
    setError(null);
    try {
      await deleteDocument(id);
      onChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed.");
    }
  }

  if (documents.length === 0) {
    return <p className="py-8 text-center text-sm text-muted-foreground">No documents yet.</p>;
  }

  return (
    <>
      {error && <p className="text-sm text-red-600">{error}</p>}
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-left text-muted-foreground">
            <th className="py-2 font-medium">Title</th>
            <th className="py-2 font-medium">Status</th>
            <th className="py-2 font-medium">Pages</th>
            <th className="py-2 font-medium">Chunks</th>
            <th className="py-2" />
          </tr>
        </thead>
        <tbody>
          {documents.map((doc) => (
            <tr key={doc.id} className="border-b">
              <td className="py-2">
                {doc.title}
                {doc.status === "failed" && doc.error_message && (
                  <p className="text-xs text-red-600">{doc.error_message}</p>
                )}
              </td>
              <td className={`py-2 ${STATUS_STYLES[doc.status]}`}>{doc.status}</td>
              <td className="py-2">{doc.page_count ?? "—"}</td>
              <td className="py-2">{doc.chunk_count}</td>
              <td className="py-2 text-right">
                <Button variant="ghost" size="sm" onClick={() => remove(doc.id)}>
                  Delete
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
