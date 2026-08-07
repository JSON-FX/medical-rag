"use client";

import { BookOpenText, Prohibit } from "@phosphor-icons/react";
import { useEffect, useRef } from "react";

import type { Turn } from "@/lib/chatReducer";
import { declineLabel } from "@/lib/copy";

/**
 * The evidence behind the selected answer, always visible.
 *
 * This panel is the point of the layout. Every answer this system gives is
 * supposed to trace back to retrieved text, and previously that text lived in
 * a `title` tooltip — present, but effectively hidden. Showing it beside the
 * answer makes the claim checkable instead of merely asserted.
 */
export default function EvidencePanel({
  turn,
  activeIndex,
  onSelect,
}: {
  turn: Turn | null;
  activeIndex: number | null;
  onSelect: (index: number) => void;
}) {
  const refs = useRef<(HTMLLIElement | null)[]>([]);

  useEffect(() => {
    if (activeIndex === null) return;
    refs.current[activeIndex]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [activeIndex, turn]);

  const sources = turn?.sources ?? [];

  return (
    <aside
      aria-label="Evidence"
      className="flex h-full flex-col border-l bg-sidebar/40"
    >
      <header className="flex items-center gap-2 border-b px-4 py-3">
        <BookOpenText size={18} className="text-primary" aria-hidden />
        <h2 className="font-[family-name:var(--font-heading)] text-sm font-semibold">Evidence</h2>
        {sources.length > 0 && (
          <span className="ml-auto rounded-full bg-accent px-2 py-0.5 text-[11px] font-medium tabular-nums text-accent-foreground">
            {sources.length}
          </span>
        )}
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {turn?.kind === "decline" ? (
          // A decline has no sources by design — the gate declined before the
          // model ran, or the model refused the context it was given. Saying
          // so is more useful than an empty list.
          <div className="px-4 py-6 text-sm">
            <Prohibit size={20} className="mb-2 text-muted-foreground" aria-hidden />
            <p className="font-medium">{declineLabel(turn.declineReason)}</p>
            <p className="mt-1 text-muted-foreground">
              No sources were cited, because nothing in your documents supported an answer.
              That is the system working, not failing.
            </p>
          </div>
        ) : sources.length === 0 ? (
          <p className="px-4 py-6 text-sm text-muted-foreground">
            Ask a question. The passages behind each answer appear here, so you can check
            them against the answer itself.
          </p>
        ) : (
          <ol className="divide-y">
            {sources.map((source, i) => {
              const active = i === activeIndex;
              return (
                <li
                  key={source.chunk_id}
                  ref={(el) => {
                    refs.current[i] = el;
                  }}
                >
                  <button
                    type="button"
                    onClick={() => onSelect(i)}
                    aria-current={active ? "true" : undefined}
                    className={`w-full px-4 py-3 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring
                      ${active ? "bg-evidence-highlight" : "hover:bg-accent/50"}`}
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className={`inline-flex h-5 min-w-5 items-center justify-center rounded px-1 text-[11px] font-semibold tabular-nums
                          ${active ? "bg-primary text-primary-foreground" : "bg-accent text-accent-foreground"}`}
                      >
                        {i + 1}
                      </span>
                      <span className="truncate text-xs font-medium">{source.title}</span>
                      <span className="ml-auto shrink-0 text-[11px] tabular-nums text-muted-foreground">
                        p. {source.page}
                      </span>
                    </div>
                    <p className="mt-2 line-clamp-6 text-xs leading-5 text-muted-foreground">
                      {source.snippet}
                    </p>
                  </button>
                </li>
              );
            })}
          </ol>
        )}
      </div>
    </aside>
  );
}
