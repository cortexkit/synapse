import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, rename, rm } from "node:fs/promises";
import { dirname, relative, resolve } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { once } from "node:events";
import { loadGatherSystemPrompt } from "../prompts/gather-system.ts";
import {
  toOpenAiMessages,
  toOpenAiTools,
  type OpenAiChatMessage,
  type OpenAiToolDefinition,
} from "../src/openai.ts";
import { GATHER_TOOLS } from "../src/tools.ts";
import type { AnthropicContentBlock, BankedRow, TrajectoryMessage } from "../src/types.ts";
import { isRecord } from "../src/utils.ts";

const TRAIN_DIR = dirname(fileURLToPath(import.meta.url));
const PACKAGE_DIR = resolve(TRAIN_DIR, "..");
const DEFAULT_INPUT = resolve(TRAIN_DIR, "../data/dataset-v1.jsonl");
const DEFAULT_OUTPUT = resolve(TRAIN_DIR, "sft-dataset.jsonl");
const DEFAULT_REPORT = resolve(TRAIN_DIR, "conversion-report.json");
const SAMPLE_SEED = "8918";
const SAMPLE_COUNT = 5;

export interface SftExample {
  messages: OpenAiChatMessage[];
  tools: OpenAiToolDefinition[];
}

interface DropRecord {
  line: number;
  reason: string;
}

interface ValidationResult {
  toolCalls: number;
  toolResults: number;
  assistantMessages: number;
  assistantContentCharacters: number;
}

interface ConvertedRecord {
  index: number;
  source: BankedRow;
  example: SftExample;
  serialized: string;
  validation: ValidationResult;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function requireBankedRow(value: unknown, line: number): BankedRow {
  if (!isRecord(value) || typeof value.request !== "string" || !Array.isArray(value.full_trajectory)) {
    throw new Error(`line ${line} is not a BankedRow with request and full_trajectory`);
  }
  if (value.full_trajectory.length === 0) throw new Error(`line ${line} has an empty trajectory`);
  return value as unknown as BankedRow;
}

function assistantText(content: string | AnthropicContentBlock[]): string {
  if (typeof content === "string") return content;
  return content
    .filter((block): block is Extract<AnthropicContentBlock, { type: "text" }> => block.type === "text")
    .map((block) => block.text)
    .join("\n");
}

export function convertRow(row: BankedRow): SftExample {
  const trajectory = row.full_trajectory as TrajectoryMessage[];
  const first = trajectory[0];
  if (!first || first.role !== "user" || typeof first.content !== "string") {
    throw new Error("trajectory must start with the rendered user request string");
  }
  if (!first.content.includes(row.request)) {
    throw new Error("rendered user request does not contain BankedRow.request verbatim");
  }

  const last = trajectory.at(-1);
  if (!last || last.role !== "assistant") throw new Error("trajectory must end with an assistant turn");
  const finalText = assistantText(last.content);
  if (finalText.length === 0) throw new Error("final assistant turn has no produced text");
  if (typeof last.content !== "string" && last.content.some((block) => block.type === "tool_use")) {
    throw new Error("final assistant turn still contains a tool_use block");
  }

  const messages = toOpenAiMessages(loadGatherSystemPrompt(), trajectory);
  const convertedFinal = messages.at(-1);
  if (!convertedFinal || convertedFinal.role !== "assistant" || convertedFinal.content !== finalText) {
    throw new Error("converted final assistant text differs from the stored trajectory emission");
  }
  return { messages, tools: toOpenAiTools(GATHER_TOOLS) };
}

export function validateOpenAiExample(example: SftExample): ValidationResult {
  if (example.tools.length !== 9) throw new Error(`expected 9 tools, found ${example.tools.length}`);
  if (example.messages[0]?.role !== "system") throw new Error("first message is not system");
  if (example.messages[1]?.role !== "user") throw new Error("second message is not user");
  if (example.messages.at(-1)?.role !== "assistant") throw new Error("last message is not assistant");

  const seenCallIds = new Set<string>();
  let pendingCallIds = new Set<string>();
  let toolCalls = 0;
  let toolResults = 0;
  let assistantMessages = 0;
  let assistantContentCharacters = 0;

  for (const [index, message] of example.messages.entries()) {
    if (index > 0 && message.role === "system") throw new Error(`message ${index} is an extra system turn`);

    if (message.role === "assistant") {
      if (pendingCallIds.size > 0) {
        throw new Error(`assistant message ${index} appears before tool results for ${[...pendingCallIds].join(", ")}`);
      }
      assistantMessages += 1;
      assistantContentCharacters += message.content?.length ?? 0;
      const calls = message.tool_calls ?? [];
      pendingCallIds = new Set<string>();
      for (const [callIndex, call] of calls.entries()) {
        if (call.type !== "function" || !call.id || !call.function.name) {
          throw new Error(`assistant message ${index} has an incomplete tool call at ${callIndex}`);
        }
        if (seenCallIds.has(call.id)) throw new Error(`duplicate tool_call id ${call.id}`);
        try {
          JSON.parse(call.function.arguments);
        } catch (error) {
          throw new Error(`tool_call ${call.id} arguments are not JSON: ${errorMessage(error)}`);
        }
        seenCallIds.add(call.id);
        pendingCallIds.add(call.id);
        toolCalls += 1;
      }
      continue;
    }

    if (message.role === "tool") {
      if (!message.tool_call_id) throw new Error(`tool message ${index} lacks tool_call_id`);
      if (!pendingCallIds.delete(message.tool_call_id)) {
        throw new Error(`tool message ${index} is orphaned or duplicated: ${message.tool_call_id}`);
      }
      toolResults += 1;
      continue;
    }

    if (pendingCallIds.size > 0) {
      throw new Error(`${message.role} message ${index} appears before tool results for ${[...pendingCallIds].join(", ")}`);
    }
  }

  if (pendingCallIds.size > 0) throw new Error(`missing tool results for ${[...pendingCallIds].join(", ")}`);
  if (toolCalls !== toolResults) throw new Error(`tool call/result mismatch: ${toolCalls}/${toolResults}`);
  const final = example.messages.at(-1)!;
  if ((final.tool_calls?.length ?? 0) > 0 || !final.content) {
    throw new Error("last assistant message must be a nonempty produced final answer without tool calls");
  }
  return { toolCalls, toolResults, assistantMessages, assistantContentCharacters };
}

function sampleRank(index: number): string {
  return createHash("sha256").update(`${SAMPLE_SEED}:${index}`).digest("hex");
}

async function sha256(path: string): Promise<string> {
  const hash = createHash("sha256");
  const stream = createReadStream(path);
  stream.on("data", (chunk) => hash.update(chunk));
  await once(stream, "end");
  return hash.digest("hex");
}

async function writeLine(stream: ReturnType<typeof createWriteStream>, line: string): Promise<void> {
  if (!stream.write(line)) await once(stream, "drain");
}

async function main(): Promise<void> {
  const input = resolve(process.argv[2] ?? DEFAULT_INPUT);
  const output = resolve(process.argv[3] ?? DEFAULT_OUTPUT);
  const reportPath = resolve(process.argv[4] ?? DEFAULT_REPORT);
  const temporaryOutput = `${output}.tmp`;
  await mkdir(dirname(output), { recursive: true });
  await rm(temporaryOutput, { force: true });

  const reader = createInterface({ input: createReadStream(input), crlfDelay: Infinity });
  const writer = createWriteStream(temporaryOutput, { encoding: "utf8" });
  const drops: DropRecord[] = [];
  const converted: ConvertedRecord[] = [];
  let rowsIn = 0;

  try {
    for await (const line of reader) {
      rowsIn += 1;
      if (line.trim().length === 0) {
        drops.push({ line: rowsIn, reason: "blank input line" });
        continue;
      }
      try {
        const source = requireBankedRow(JSON.parse(line), rowsIn);
        const example = convertRow(source);
        const validation = validateOpenAiExample(example);
        const serialized = JSON.stringify(example);
        await writeLine(writer, `${serialized}\n`);
        converted.push({ index: rowsIn - 1, source, example, serialized, validation });
      } catch (error) {
        drops.push({ line: rowsIn, reason: errorMessage(error) });
      }
    }
  } finally {
    writer.end();
    await once(writer, "finish");
  }

  await rename(temporaryOutput, output);
  const dropRate = rowsIn === 0 ? 1 : drops.length / rowsIn;
  const samples = [...converted]
    .sort((left, right) => sampleRank(left.index).localeCompare(sampleRank(right.index)))
    .slice(0, SAMPLE_COUNT);
  if (samples.length !== SAMPLE_COUNT) throw new Error(`need ${SAMPLE_COUNT} converted rows for random verification`);
  const sideBySide = [...samples].sort((left, right) => left.serialized.length - right.serialized.length)[0]!;

  const report = {
    generated_at: new Date().toISOString(),
    input: relative(PACKAGE_DIR, input),
    output: relative(PACKAGE_DIR, output),
    input_sha256: await sha256(input),
    output_sha256: await sha256(output),
    rows_in: rowsIn,
    rows_out: converted.length,
    rows_dropped: drops.length,
    drop_rate: dropRate,
    drop_reasons: Object.entries(
      drops.reduce<Record<string, number>>((counts, drop) => {
        counts[drop.reason] = (counts[drop.reason] ?? 0) + 1;
        return counts;
      }, {}),
    ).map(([reason, count]) => ({ reason, count })),
    dropped_rows: drops,
    tools: {
      count: toOpenAiTools(GATHER_TOOLS).length,
      names: GATHER_TOOLS.map((tool) => tool.name),
      schema_sha256: createHash("sha256").update(JSON.stringify(toOpenAiTools(GATHER_TOOLS))).digest("hex"),
    },
    random_verification: {
      method: `five smallest SHA-256 ranks of ${SAMPLE_SEED}:zero-based-row-index`,
      samples: samples.map((sample) => ({
        row_index: sample.index,
        repo_full: sample.source.repo_full,
        request: sample.source.request,
        messages: sample.example.messages.length,
        ...sample.validation,
        final_text_sha256: createHash("sha256")
          .update(String(sample.example.messages.at(-1)!.content))
          .digest("hex"),
        passed: true,
      })),
    },
    side_by_side: {
      row_index: sideBySide.index,
      note: "The original trajectory and complete converted training example below are intentionally unabridged.",
      anthropic_original: sideBySide.source.full_trajectory,
      openai_converted: sideBySide.example,
    },
    argument_fidelity:
      "Anthropic input blocks in BankedRow are parsed JSON values, not raw response substrings. The converter therefore reuses src/openai.ts, whose toolArguments applies the only possible compact JSON.stringify serialization while preserving IDs, names, values, key insertion order, strings, and numeric values.",
  };
  await Bun.write(reportPath, `${JSON.stringify(report, null, 2)}\n`);

  console.log(
    JSON.stringify({
      input,
      output,
      report: reportPath,
      rowsIn,
      rowsOut: converted.length,
      rowsDropped: drops.length,
      dropRate,
      sampleIndices: samples.map((sample) => sample.index),
      inputSha256: report.input_sha256,
      outputSha256: report.output_sha256,
    }),
  );
  if (dropRate > 0.05) throw new Error(`conversion dropped ${(dropRate * 100).toFixed(2)}%, exceeding the 5% stop threshold`);
}

if (import.meta.main) await main();
