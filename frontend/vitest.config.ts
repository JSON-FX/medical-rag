import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Node, not jsdom: only pure modules under lib/ are tested. Component
    // tests are deliberately out of scope (spec 7).
    environment: "node",
    include: ["lib/**/*.test.ts"],
  },
});
