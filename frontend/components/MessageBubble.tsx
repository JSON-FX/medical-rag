"use client";

import { Prohibit, WarningOctagon } from "@phosphor-icons/react";

import AnswerText from "@/components/AnswerText";
import type { Turn } from "@/lib/chatReducer";
import { declineLabel, errorRecovery } from "@/lib/copy";

export default function MessageBubble({
  turn,
  selected,
  activeIndex,
  onCite,
  onSelect,
}: {
  turn: Turn;
  selected: boolean;
  activeIndex: number | null;
  onCite: (index: number) => void;
  onSelect: () => void;
}) {
  if (turn.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-primary px-4 py-2.5 text-sm text-primary-foreground">
          {turn.text}
        </div>
      </div>
    );
  }

  if (turn.kind === "error") {
    return (
      <div className="rounded-xl border border-destructive/30 bg-destructive/10 p-4">
        <div className="flex items-center gap-2 text-destructive">
          <WarningOctagon size={18} weight="fill" aria-hidden />
          <p className="text-sm font-semibold">Something went wrong</p>
        </div>
        <p className="mt-1.5 text-sm">
          {turn.errorMessage ?? errorRecovery(turn.errorCode ?? "")}
        </p>
        {/* Partial text is kept rather than discarded: a stream that dies
            halfway still produced something the reader may want. */}
        {turn.text && (
          <p className="mt-3 whitespace-pre-wrap border-t border-destructive/20 pt-3 text-sm text-muted-foreground">
            {turn.text}
          </p>
        )}
      </div>
    );
  }

  if (turn.kind === "decline") {
    return (
      <div className="rounded-xl border border-decline-border bg-decline p-4">
        <div className="flex items-center gap-2">
          <Prohibit size={16} className="text-decline-foreground" aria-hidden />
          <p className="text-[11px] font-semibold uppercase tracking-wide text-decline-foreground">
            {declineLabel(turn.declineReason)}
          </p>
        </div>
        {/* Server-authored copy, rendered verbatim. */}
        <p className="mt-2 text-sm leading-6">{turn.text}</p>
      </div>
    );
  }

  if (turn.kind === "pending" && !turn.text) {
    return (
      <p className="flex items-center gap-2 text-sm text-muted-foreground">
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-primary" aria-hidden />
        Searching your documents…
      </p>
    );
  }

  return (
    <div
      onClick={onSelect}
      className={`rounded-xl border bg-card p-4 text-sm transition-colors ${
        selected ? "border-primary/40" : "border-border"
      }`}
    >
      <AnswerText
        text={turn.text}
        sourceCount={turn.sources.length}
        activeIndex={selected ? activeIndex : null}
        onCite={onCite}
      />
      {turn.sources.length > 0 && (
        <p className="mt-3 border-t pt-2.5 text-xs text-muted-foreground">
          Grounded in {turn.sources.length} passage{turn.sources.length > 1 ? "s" : ""} —
          select a number to see it.
        </p>
      )}
      {turn.truncated && (
        <p className="mt-2 text-xs text-destructive">
          This answer was cut off before it finished.
        </p>
      )}
    </div>
  );
}
