"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { getHealth } from "@/lib/api";
import type { HealthReport } from "@/lib/types";

type Problem = { tone: "error" | "warning"; text: React.ReactNode };

/**
 * Each condition has a different fix, so each gets its own message.
 *
 * The empty-corpus case matters most: with no documents every question
 * declines with `empty_corpus`, which reads as a broken app rather than an
 * empty one unless the UI says so.
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
  ].filter(Boolean) as string[];

  if (missing.length > 0) {
    return {
      tone: "error",
      text: (
        <>
          Missing {missing.length === 1 ? "model" : "models"}. Run{" "}
          {missing.map((model, i) => (
            <span key={model}>
              {i > 0 ? " and " : ""}
              <code className="rounded bg-black/10 px-1">ollama pull {model}</code>
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
          <Link href="/documents" className="underline">
            Upload one
          </Link>
          .
        </>
      ),
    };
  }
  return null;
}

export default function HealthBanner() {
  const [problem, setProblem] = useState<Problem | null>(null);

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
  }, []);

  if (!problem) return null;

  const tone =
    problem.tone === "error"
      ? "bg-red-50 text-red-900 dark:bg-red-950 dark:text-red-100"
      : "bg-blue-50 text-blue-900 dark:bg-blue-950 dark:text-blue-100";

  return (
    <div role="status" className={`border-b px-4 py-2 text-center text-sm ${tone}`}>
      {problem.text}
    </div>
  );
}
