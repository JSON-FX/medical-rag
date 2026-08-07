import { declineLabel } from "@/lib/copy";

/**
 * A decline is the system working, so it reads as an explanation rather than
 * an error. The body is server-authored copy rendered verbatim (spec 5.1);
 * only the label above it is chosen here.
 */
export default function DeclineCard({
  text,
  reason,
}: {
  text: string;
  reason: string | null;
}) {
  return (
    <div className="rounded-lg border border-dashed bg-muted/40 p-3">
      <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
        {declineLabel(reason)}
      </p>
      <p className="text-sm">{text}</p>
    </div>
  );
}
