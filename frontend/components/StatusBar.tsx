"use client";

import { Info, WarningCircle, WarningOctagon } from "@phosphor-icons/react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { getHealth } from "@/lib/api";
import type { HealthReport } from "@/lib/types";

const DISCLAIMER =
  "For informational reference only — not a substitute for professional medical judgment.";

type Tone = "error" | "warning";
type Problem = { tone: Tone; text: React.ReactNode };

/**
 * Each condition has a different fix, so each keeps its own message.
 *
 * The empty-corpus case matters most: with no documents every question is
 * declined by the gate, which reads as a broken app rather than an empty one
 * unless the UI says otherwise.
 */
function diagnose(health: HealthReport): Problem | null {
  if (!health.ollama_reachable) {
    return {
      tone: "error",
      text: <>Ollama isn&apos;t reachable at {health.host}. Start it and reload.</>,
    };
  }
  const missing = [
    !health.models.chat ? health.expected.chat : null,
    !health.models.embed ? health.expected.embed : null,
  ].filter((m): m is string => m !== null);

  if (missing.length > 0) {
    return {
      tone: "error",
      text: (
        <>
          Missing {missing.length === 1 ? "model" : "models"}. Run{" "}
          {missing.map((model, i) => (
            <span key={model}>
              {i > 0 ? " and " : ""}
              <code className="rounded bg-black/10 px-1 dark:bg-white/15">
                ollama pull {model}
              </code>
            </span>
          ))}
          .
        </>
      ),
    };
  }
  if (health.documents_ready === 0) {
    return {
      tone: "warning",
      text: (
        <>
          No documents uploaded yet, so every question will be declined.{" "}
          <Link href="/documents" className="font-medium underline underline-offset-2">
            Upload one
          </Link>
          .
        </>
      ),
    };
  }
  return null;
}

export default function StatusBar() {
  const [problem, setProblem] = useState<Problem | null>(null);
  const pathname = usePathname();

  // Keyed on pathname: the root layout is preserved across client-side
  // navigation, so a mount-only effect would never re-run — leaving the
  // banner asserting "no documents" right after a successful upload.
  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((health) => {
        if (!cancelled) setProblem(diagnose(health));
      })
      .catch(() => {
        if (!cancelled) {
          setProblem({
            tone: "error",
            text: <>Can&apos;t reach the backend. Is Django running on port 8000?</>,
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [pathname]);

  const Icon =
    problem?.tone === "error" ? WarningOctagon : problem ? WarningCircle : Info;

  const tone = !problem
    ? "bg-muted/60 text-muted-foreground"
    : problem.tone === "error"
      ? "bg-destructive/10 text-destructive"
      : "bg-primary/10 text-accent-foreground";

  return (
    <div
      // role=alert only when something is actually wrong, so the disclaimer
      // does not interrupt a screen reader on every navigation.
      role={problem?.tone === "error" ? "alert" : "status"}
      className={`flex flex-wrap items-center gap-x-3 gap-y-1 border-b px-4 py-2 text-xs ${tone}`}
    >
      <Icon size={15} weight="fill" className="shrink-0" aria-hidden />
      <span>{problem ? problem.text : DISCLAIMER}</span>
      {problem && (
        <span className="ml-auto hidden text-muted-foreground sm:inline">{DISCLAIMER}</span>
      )}
    </div>
  );
}
