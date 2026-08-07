"use client";

import { useCallback, useEffect, useState } from "react";

import DocumentTable from "@/components/DocumentTable";
import DocumentUploader from "@/components/DocumentUploader";
import { listDocuments } from "@/lib/api";
import type { DocumentRecord } from "@/lib/types";

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    listDocuments()
      .then((records) => {
        setDocuments(records);
        // Clearing on success matters because refresh() is called after every
        // upload and delete: without it a "Failed to fetch" from one blip
        // sits above a list that is visibly working again.
        setError(null);
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Could not load documents."));
  }, []);

  useEffect(refresh, [refresh]);

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-xl font-semibold">Documents</h1>
      <DocumentUploader onUploaded={refresh} />
      {error && <p className="text-sm text-red-600">{error}</p>}
      <DocumentTable documents={documents} onChanged={refresh} />
    </div>
  );
}
