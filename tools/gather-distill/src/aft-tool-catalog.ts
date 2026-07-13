// Generated from cortexkit/aft v0.46.0 subc_tool_schemas.json.
// Keep this catalog pinned to the AFT campaign binary.
export const PRODUCTION_AFT_TOOL_SCHEMAS = {
  "search": {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
      "query": {
        "description": "Concept, regex, literal text, filename, or capability to find. Examples: 'fuzzy match with whitespace tolerance', '^export', 'Cargo.lock'.",
        "type": "string"
      },
      "topK": {
        "description": "Number of results (default: 10, max: 100)",
        "type": "integer",
        "minimum": 1,
        "maximum": 100
      },
      "hint": {
        "description": "Optional routing hint. Defaults to 'auto'.",
        "type": "string",
        "enum": [
          "regex",
          "literal",
          "semantic",
          "auto"
        ]
      },
      "includeTests": {
        "description": "Include test files (*.test.*, *_test.rs, __tests__/, …) plus test-support, fixture, mock, snapshot, and corpus files. Defaults to false.",
        "type": "boolean"
      },
      "path": {
        "description": "Search a different project root (absolute or ~ path). Requires that project to have been indexed by AFT.",
        "type": "string"
      }
    },
    "required": [
      "query"
    ],
    "description": "Search code with one tool: concepts, identifiers, error strings, regex, literals, and filenames are auto-routed to the right engine and returned ranked. For conceptual 'how does X work' queries, phrase a full natural-language sentence — the semantic lane is NL-aware and matches intent against docstrings and comments ('how does the ORM build and execute a query', 'where is rate limiting handled'), not just keywords. Exact names, strings, and regex stay terse ('^export', 'Cargo.lock').\n\nSet hint to 'regex', 'literal', or 'semantic' to force a lane."
  },
  "outline": {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
      "target": {
        "description": "What to outline: a file path, directory path, URL, or array of paths. The mode is auto-detected: URLs by `http://`/`https://` prefix, directories by stat, arrays as multi-file.",
        "anyOf": [
          {
            "type": "string"
          },
          {
            "type": "array",
            "items": {
              "type": "string"
            }
          }
        ]
      },
      "files": {
        "description": "Directory-only mode: when true, target must be a directory or array of directories and the result is a flat file tree with path, language, symbol count, and byte size instead of a symbol outline.",
        "type": "boolean"
      },
      "includeTests": {
        "description": "Directory outline only: include test files. Defaults to false; tests are hidden.",
        "type": "boolean"
      }
    },
    "required": [
      "target"
    ],
    "description": "Structural outline of source code, documentation files, or remote URLs. For code, returns symbols (functions, classes, types) with line ranges. For Markdown and HTML, returns heading hierarchy. Use this to explore structure before reading specific sections with aft_zoom. Set `files: true` with a directory target for a flat indexed file tree with language, symbol count, and byte metadata.\n\nFor understanding a specific feature, prefer aft_search + aft_zoom on named symbols; use aft_outline on a whole directory only for high-level structure mapping. aft_zoom with `callgraph:true` gives one-level forward calls-out; use aft_callgraph only for reverse callers or multi-level traces.\n\nPass a single `target`:\n  • file path → outline that file (with signatures)\n  • directory path → outline all source files under it (recursively, up to 200 files)\n  • URL (http:// or https://) → fetch and outline a remote HTML/Markdown document\n  • array of paths → outline multiple files in one call; with files:true, every path must be a directory"
  },
  "zoom": {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
      "filePath": {
        "description": "Path to file (absolute or relative to project root)",
        "type": "string"
      },
      "url": {
        "description": "HTTP/HTTPS URL of an HTML or Markdown document to fetch and zoom into",
        "type": "string"
      },
      "symbols": {
        "description": "Symbol name for code, or heading text for Markdown/HTML. Pass a string for one lookup or an array for batched lookups in the same file/URL.",
        "anyOf": [
          {
            "type": "string"
          },
          {
            "type": "array",
            "items": {
              "type": "string"
            }
          }
        ]
      },
      "targets": {
        "description": "Cross-file batch: `{ filePath, symbol }` or an array of them. Mutually exclusive with filePath/url/symbols.",
        "anyOf": [
          {
            "type": "object",
            "properties": {
              "filePath": {
                "description": "Path to file (absolute or relative to project root)",
                "type": "string"
              },
              "symbol": {
                "description": "Symbol name in that file",
                "type": "string"
              }
            },
            "required": [
              "filePath",
              "symbol"
            ]
          },
          {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "filePath": {
                  "description": "Path to file (absolute or relative to project root)",
                  "type": "string"
                },
                "symbol": {
                  "description": "Symbol name in that file",
                  "type": "string"
                }
              },
              "required": [
                "filePath",
                "symbol"
              ]
            }
          }
        ]
      },
      "contextLines": {
        "description": "Lines of context before/after the symbol (default: 3)",
        "type": "integer",
        "minimum": 1,
        "maximum": 9007199254740991
      },
      "callgraph": {
        "description": "Include call-graph annotations (calls-out / called-by within the same file). Default false; off keeps zoom output minimal.",
        "type": "boolean"
      }
    },
    "description": "Inspect code symbols or documentation sections. For code, returns the full source of a symbol. Pass `callgraph: true` to also include call-graph annotations (calls-out / called-by within the same file). For Markdown and HTML, returns the section content under the given heading.\n\nUse exactly ONE mode: `{ filePath, symbols }`, `{ url, symbols }`, or `{ targets }`. `symbols` can be a string or array (one or many lookups in the same file/URL). Use `targets` for cross-file batches: `{ filePath, symbol }` or an array of them."
  },
  "callgraph": {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
      "op": {
        "description": "Navigation operation",
        "type": "string",
        "enum": [
          "call_tree",
          "callers",
          "trace_to",
          "trace_to_symbol",
          "impact",
          "trace_data"
        ]
      },
      "filePath": {
        "description": "Path to the source file containing the symbol (absolute or relative to project root)",
        "type": "string"
      },
      "symbol": {
        "description": "Name of the symbol to analyze",
        "type": "string"
      },
      "depth": {
        "description": "Max traversal depth (default: call_tree=5, callers=1, trace_to=10, trace_to_symbol=10 capped at 16, impact=5, trace_data=5)",
        "type": "integer",
        "minimum": 1,
        "maximum": 9007199254740991
      },
      "expression": {
        "description": "Expression to track through data flow (required for trace_data op)",
        "type": "string"
      },
      "toSymbol": {
        "description": "Target symbol name for trace_to_symbol; the returned path ends at this symbol",
        "type": "string"
      },
      "toFile": {
        "description": "Optional target file for trace_to_symbol; required when toSymbol exists in multiple files",
        "type": "string"
      },
      "includeTests": {
        "description": "Include test files in callers/paths. Defaults to false; tests are hidden.",
        "type": "boolean"
      },
      "includeUnresolved": {
        "description": "Show every unresolved external/stdlib call individually. Defaults to false; unresolved leaf calls are collapsed into one summary per parent.",
        "type": "boolean"
      }
    },
    "required": [
      "op",
      "filePath",
      "symbol"
    ],
    "description": "Answer code-relationship questions from a real call graph — instead of grep + read chains. Reach for this whenever the question is about how symbols connect: who calls X, what X calls, what breaks if X changes, how execution reaches X, or how a value flows.\n\nUse aft_zoom with `callgraph:true` for one-level forward calls-out while reading source. Use aft_callgraph only for reverse callers or multi-level traces so you do not double-fetch the same relationships.\n\nOps:\n- 'callers': Find all call sites of a symbol. Use before renaming or changing a function's signature.\n- 'impact': What breaks if a symbol changes — affected callers with signatures and entry-point status (blast radius). Use before a risky edit.\n- 'call_tree': What a function calls (forward traversal). Use to understand a function's dependencies before modifying it.\n- 'trace_to': How execution reaches a function from entry points (routes, exports, main). Use to understand context around deeply-nested code.\n- 'trace_to_symbol': Shortest call path from one symbol to another. Requires 'toSymbol'. If multiple targets match, the error returns candidate files; retry with 'toFile' to disambiguate.\n- 'trace_data': Follow a value through variable assignments and function parameters across files. Requires 'symbol' (scope to trace from) and 'expression'.\n\nAll ops require both 'filePath' and 'symbol'. 'expression' is additionally required for trace_data; 'toSymbol' for trace_to_symbol.\n\nMarkers: ~ = edge resolved by name only (may point at the wrong same-named symbol); [unresolved] = callee not resolved to a definition, so the location shown is the call site. Unmarked edges are resolved exactly. By default, unresolved external/stdlib leaf calls in call_tree are collapsed into one summary per parent; pass includeUnresolved=true to show every unresolved edge individually.\n"
  },
  "read": {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
      "filePath": {
        "description": "Path to file or directory (absolute or relative to project root)",
        "type": "string"
      },
      "startLine": {
        "description": "1-based line to start reading from",
        "type": "integer",
        "minimum": 1,
        "maximum": 9007199254740991
      },
      "endLine": {
        "description": "1-based line to stop reading at (inclusive)",
        "type": "integer",
        "minimum": 1,
        "maximum": 9007199254740991
      },
      "limit": {
        "description": "Max lines to return (default: 2000)",
        "type": "integer",
        "minimum": 1,
        "maximum": 9007199254740991
      },
      "offset": {
        "description": "1-based line number to start reading from (use with limit). Ignored if startLine is provided",
        "type": "integer",
        "minimum": 1,
        "maximum": 9007199254740991
      }
    },
    "required": [
      "filePath"
    ],
    "description": "Read file contents or list directory entries.\n\nUse either startLine/endLine OR offset/limit to read a section of a file.\n\nBehavior:\n- Returns line-numbered content (e.g., \"1: const x = 1\")\n- Lines longer than 2000 characters are truncated\n- Output capped at 50KB\n- Binary files are auto-detected and return a size-only message\n- Supported images (PNG, JPEG, GIF, WebP) and PDFs are returned as tool attachments; range arguments are ignored for media\n- Directories return sorted entries with trailing / for subdirectories\n\nExamples:\n  Read full file: { \"filePath\": \"src/app.ts\" }\n  Read lines 50-100: { \"filePath\": \"src/app.ts\", \"startLine\": 50, \"endLine\": 100 }\n  Read 30 lines from line 200: { \"filePath\": \"src/app.ts\", \"offset\": 200, \"limit\": 30 }\n  List directory: { \"filePath\": \"src/\" }\n"
  },
  "grep": {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
      "pattern": {
        "type": "string"
      },
      "include": {
        "type": "string"
      },
      "path": {
        "type": "string"
      }
    },
    "required": [
      "pattern"
    ],
    "description": "Search file contents using regular expressions. Returns matching lines with file paths and line numbers (no surrounding context lines — use `read` for that). Always case-sensitive. Capped at 100 matches; if you hit the cap, narrow with `path` or `include` and re-run."
  },
  "glob": {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
      "pattern": {
        "type": "string"
      },
      "path": {
        "type": "string"
      }
    },
    "required": [
      "pattern"
    ],
    "description": "Find files matching a glob pattern. Returns matching file paths sorted by modification time."
  },
  "inspect": {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
      "sections": {
        "description": "Categories to include in detailed drill-down (e.g. 'todos' or ['todos', 'dead_code', 'cycles']). Use 'all' for every active category. Omit for summary-only mode.",
        "anyOf": [
          {
            "type": "string"
          },
          {
            "type": "array",
            "items": {
              "type": "string"
            }
          }
        ]
      },
      "scope": {
        "description": "Restrict scan/results to paths under this scope (file or directory, absolute or relative to project root). Tier 1 scopes the scan; Tier 2 scans project-wide and applies scope as a result filter.",
        "anyOf": [
          {
            "type": "string"
          },
          {
            "type": "array",
            "items": {
              "type": "string"
            }
          }
        ]
      },
      "topK": {
        "description": "Max drill-down items per category. Default 20, max 100.",
        "type": "integer",
        "exclusiveMinimum": 0,
        "maximum": 100
      }
    },
    "description": "Codebase health snapshot. One call returns summary stats for: TODOs, diagnostics, file/symbol metrics, dead code, unused exports, code duplicates, and TS/JS import cycles. Pass `sections` for per-category drill-down details.\n\nCategories run in tiers — Tier 1 (todos, metrics) return synchronously from cache. Tier 2 (dead_code, unused_exports, duplicates, cycles) waits for a fresh reuse scan up to a short deadline; if a category is still scanning the response reports `complete: false` with `pending_categories: [...]` rather than a fabricated clean count. Rust module cycles are out of scope for `cycles`.\n\nUse when: starting work on unfamiliar code, after multi-edit batches to check diagnostics, before a refactor, before review, or to verify cleanup completeness.\n\nTreat `dead_code` as a hint, not proof: reachability is call-based, so symbols reached only via method dispatch or referenced only in type position may be false positives — verify before deleting."
  },
  "conflicts": {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
      "path": {
        "description": "Optional path inside the git repository or worktree to inspect (absolute or relative to project root). Conflicts are discovered from that repository's top level. Defaults to the session project root.",
        "type": "string"
      }
    },
    "description": "Show all git merge conflicts across the repository — returns line-numbered conflict regions with context for every conflicted file in a single call. Conflicts are discovered from the git repository's top level. By default it inspects the session's project repository; pass `path` to inspect a different repository or git worktree (e.g. where a rebase/merge is running)."
  }
} as const;
