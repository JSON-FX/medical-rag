import MessageBubble from "@/components/MessageBubble";
import type { Turn } from "@/lib/chatReducer";

export default function MessageList({ turns }: { turns: Turn[] }) {
  if (turns.length === 0) {
    return (
      <p className="py-12 text-center text-sm text-muted-foreground">
        Ask a question about the documents you&apos;ve uploaded.
      </p>
    );
  }
  return (
    <div className="flex flex-col gap-4">
      {turns.map((turn, i) => (
        <MessageBubble key={i} turn={turn} />
      ))}
    </div>
  );
}
