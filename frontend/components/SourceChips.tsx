import type { Source } from "@/lib/types";

export default function SourceChips({ sources }: { sources: Source[] }) {
  if (sources.length === 0) return null;
  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      {sources.map((source) => (
        <span
          key={source.chunk_id}
          title={source.snippet}
          className="rounded-full border bg-muted px-2 py-0.5 text-xs text-muted-foreground"
        >
          {source.title} · p.{source.page}
        </span>
      ))}
    </div>
  );
}
