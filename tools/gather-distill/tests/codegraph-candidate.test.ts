import { expect, test } from "bun:test";
import { adaptExploreResponse, scopePathsFromExploreMarkdown, snippetsFromExploreMarkdown } from "../scripts/codegraph-candidate.ts";

test("maps CodeGraph numbered source blocks without bridging omitted lines", () => {
  const markdown = [
    "**Exploration: trace the feature**",
    "",
    "**Source Code**",
    "",
    "**`src/feature.ts`** — primary match: runFeature",
    "```typescript",
    "10\texport function runFeature() {",
    "11\t  return helper();",
    "12\t}",
    "20\tfunction helper() {",
    "21\t  return true;",
    "22\t}",
    "```",
  ].join("\n");

  expect(snippetsFromExploreMarkdown(markdown)).toEqual([
    { path: "src/feature.ts", startLine: 10, endLine: 12, why: "codegraph_explore primary match: runFeature" },
    { path: "src/feature.ts", startLine: 20, endLine: 22, why: "codegraph_explore primary match: runFeature" },
  ]);
  expect(scopePathsFromExploreMarkdown(`${markdown}\n- caller @src/caller.ts:9`)).toEqual(["src/feature.ts", "src/caller.ts"]);
});

test("uses the verbatim request and rejects source-less explore responses", () => {
  const request = "Where does the feature start?";
  const adapted = adaptExploreResponse(request, {
    content: [{ type: "text", text: "**Exploration: no indexed code**" }],
  });
  expect(adapted.package_).toBeNull();
  expect(adapted.error).toContain("no source blocks");

  const withSource = adaptExploreResponse(request, {
    content: [{ type: "text", text: "**`src/start.ts`** — explore primary match\n```ts\n4\texport const start = true;\n```" }],
  });
  expect(withSource.package_).toEqual({
    interpretation: request,
    scope: ["src/start.ts"],
    snippets: [{ path: "src/start.ts", startLine: 4, endLine: 4, why: "codegraph_explore explore primary match" }],
    omissions: [],
  });
});
