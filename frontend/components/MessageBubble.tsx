import DeclineCard from "@/components/DeclineCard";
import SourceChips from "@/components/SourceChips";
import { errorRecovery } from "@/lib/copy";
import type { Turn } from "@/lib/chatReducer";

export default function MessageBubble({ turn }: { turn: Turn }) {
  if (turn.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-lg bg-primary px-3 py-2 text-sm text-primary-foreground">
          {turn.text}
        </div>
      </div>
    );
  }

  if (turn.kind === "error") {
    return (
      <div className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-100">
        {errorRecovery(turn.errorCode ?? "")}
      </div>
    );
  }

  if (turn.kind === "decline") {
    return <DeclineCard text={turn.text} reason={turn.declineReason} />;
  }

  if (turn.kind === "pending" && !turn.text) {
    return <p className="text-sm text-muted-foreground">Searching your documents…</p>;
  }

  return (
    <div className="max-w-[85%]">
      <div className="whitespace-pre-wrap rounded-lg border bg-card p-3 text-sm">{turn.text}</div>
      <SourceChips sources={turn.sources} />
      {turn.truncated && (
        <p className="mt-1 text-xs text-amber-700 dark:text-amber-400">
          This answer was cut off before it finished.
        </p>
      )}
    </div>
  );
}
