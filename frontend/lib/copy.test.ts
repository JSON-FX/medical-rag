import { expect, test } from "vitest";

import { declineLabel, errorRecovery } from "./copy";

test("every decline reason the server emits gets its own label", () => {
  // These four are the complete set in backend/rag/prompts.py DECLINE_COPY.
  const labels = [
    "empty_corpus",
    "off_domain",
    "weak_unsupported",
    "insufficient_context",
  ].map(declineLabel);
  expect(new Set(labels).size).toBe(4);
  expect(labels.every((label) => label.length > 0)).toBe(true);
});

test("an unknown or absent decline reason falls back rather than showing a raw key", () => {
  expect(declineLabel(null)).toBe("Can't answer from your documents");
  expect(declineLabel("something_new")).toBe("Can't answer from your documents");
});

test("each error code the server emits gets its own recovery text", () => {
  expect(errorRecovery("ollama_unavailable")).not.toBe(errorRecovery("model_missing"));
  expect(errorRecovery("unrecognised")).toBe("Something went wrong reaching the backend.");
});
