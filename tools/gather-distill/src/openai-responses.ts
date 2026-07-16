import type { MessageRequest, MessageResponse } from "./anthropic.ts";
import type { AnthropicContentBlock, TrajectoryMessage } from "./types.ts";
import type { ToolDeclaration } from "./tools.ts";
import { isRecord } from "./utils.ts";

export interface OpenAiResponsesRequestOptions {
  /** Reasoning models on the Responses API reject temperature; omit it by default. */
  includeTemperature?: boolean;
}

function jsonArguments(input: unknown): string {
  return JSON.stringify(input ?? {}) ?? "{}";
}

function textBlocks(content: AnthropicContentBlock[]): string {
  return content
    .filter((block): block is Extract<AnthropicContentBlock, { type: "text" }> => block.type === "text")
    .map((block) => block.text)
    .join("\n");
}

/** Convert the judge trajectory to the Responses API input-item protocol. */
export function toOpenAiResponsesInput(messages: TrajectoryMessage[]): Array<Record<string, unknown>> {
  const input: Array<Record<string, unknown>> = [];
  for (const message of messages) {
    if (typeof message.content === "string") {
      input.push({ role: message.role, content: [{ type: message.role === "assistant" ? "output_text" : "input_text", text: message.content }] });
      continue;
    }

    if (message.role === "assistant") {
      const text = textBlocks(message.content);
      if (text) input.push({ role: "assistant", content: [{ type: "output_text", text }] });
      for (const block of message.content) {
        if (block.type === "tool_use") {
          input.push({
            type: "function_call",
            call_id: block.id,
            name: block.name,
            arguments: jsonArguments(block.input),
          });
        }
      }
      continue;
    }

    const text = textBlocks(message.content);
    if (text) input.push({ role: "user", content: [{ type: "input_text", text }] });
    for (const block of message.content) {
      if (block.type === "tool_result") {
        input.push({ type: "function_call_output", call_id: block.tool_use_id, output: block.content });
      }
    }
  }
  return input;
}

/** Map the existing nine AFT declarations to Responses function tools. */
export function toOpenAiResponsesTools(tools: ToolDeclaration[]): Array<Record<string, unknown>> {
  return tools.map((tool) => ({
    type: "function",
    name: tool.name,
    description: tool.description,
    parameters: tool.input_schema,
    strict: false,
  }));
}

export function openAiResponsesBody(
  request: MessageRequest,
  options: OpenAiResponsesRequestOptions = {},
): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model: request.model,
    instructions: request.system,
    input: toOpenAiResponsesInput(request.messages),
    store: false,
    stream: true,
  };
  if (request.tools !== undefined) body.tools = toOpenAiResponsesTools(request.tools);
  if (request.tool_choice) body.tool_choice = "none";
  if (options.includeTemperature && request.temperature !== undefined) body.temperature = request.temperature;
  // OpenCode's Responses path uses a medium reasoning budget and automatic
  // summaries for GPT reasoning models. Temperature is intentionally omitted.
  body.reasoning = { effort: "medium", summary: "auto" };
  return body;
}

function parseResponseText(value: unknown): string {
  if (typeof value === "string") return value;
  if (!Array.isArray(value)) return "";
  return value
    .flatMap((part) => (isRecord(part) && part.type === "output_text" && typeof part.text === "string" ? [part.text] : []))
    .join("\n");
}

function responseUsage(value: Record<string, unknown>): { input: number; output: number; thinking: number } {
  if (!isRecord(value.usage)) throw new Error("OpenAI Responses response is missing usage");
  const input = value.usage.input_tokens;
  const output = value.usage.output_tokens;
  if (typeof input !== "number" || typeof output !== "number") {
    throw new Error("OpenAI Responses response is missing token usage");
  }
  const details = isRecord(value.usage.output_tokens_details) ? value.usage.output_tokens_details : undefined;
  const reasoning = details?.reasoning_tokens;
  return { input, output, thinking: typeof reasoning === "number" && reasoning >= 0 ? reasoning : 0 };
}

/** Read the terminal response object from the Responses SSE protocol. */
export async function readOpenAiResponsesStream(response: Response): Promise<unknown> {
  const raw = await response.text();
  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.includes("text/event-stream") && !/^\s*(?:event:|data:)/u.test(raw)) {
    try {
      return JSON.parse(raw);
    } catch (error) {
      throw new Error(`OpenAI Responses returned neither JSON nor SSE: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  let event = "";
  let data: string[] = [];
  let completed: Record<string, unknown> | undefined;
  const outputItems = new Map<number, Record<string, unknown>>();

  const appendTextDelta = (index: number, delta: string): void => {
    const item = outputItems.get(index) ?? { type: "message", role: "assistant", content: [] };
    const content = Array.isArray(item.content) ? item.content as Array<Record<string, unknown>> : [];
    const existing = content.find((part) => part.type === "output_text");
    if (existing && typeof existing.text === "string") existing.text += delta;
    else content.push({ type: "output_text", text: delta });
    item.content = content;
    outputItems.set(index, item);
  };

  const consume = () => {
    if (data.length === 0) return;
    const text = data.join("\n");
    data = [];
    if (text === "[DONE]") return;
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch (error) {
      throw new Error(`OpenAI Responses SSE event is invalid JSON: ${error instanceof Error ? error.message : String(error)}`);
    }
    if (!isRecord(parsed)) return;
    const type = typeof parsed.type === "string" ? parsed.type : event;
    if (type === "error") {
      const message = isRecord(parsed.error) && typeof parsed.error.message === "string" ? parsed.error.message : "Responses stream failed";
      throw new Error(message);
    }
    if (type === "response.output_item.added" && typeof parsed.output_index === "number" && isRecord(parsed.item)) {
      outputItems.set(parsed.output_index, { ...parsed.item });
    } else if (type === "response.output_item.done" && typeof parsed.output_index === "number" && isRecord(parsed.item)) {
      outputItems.set(parsed.output_index, { ...parsed.item });
    } else if (type === "response.function_call_arguments.delta" && typeof parsed.output_index === "number" && typeof parsed.delta === "string") {
      const item = outputItems.get(parsed.output_index) ?? { type: "function_call", arguments: "" };
      item.arguments = `${typeof item.arguments === "string" ? item.arguments : ""}${parsed.delta}`;
      outputItems.set(parsed.output_index, item);
    } else if (type === "response.output_text.delta" && typeof parsed.output_index === "number" && typeof parsed.delta === "string") {
      appendTextDelta(parsed.output_index, parsed.delta);
    }
    if ((event === "response.completed" || event === "response.done" || type === "response.completed" || type === "response.done") && isRecord(parsed.response)) {
      completed = { ...parsed.response };
    }
  };
  for (const line of raw.split(/\r?\n/u)) {
    if (line.startsWith("event:")) event = line.slice("event:".length).trim();
    else if (line.startsWith("data:")) data.push(line.slice("data:".length).trimStart());
    else if (line.trim() === "") consume();
  }
  consume();
  if (!completed) throw new Error("OpenAI Responses SSE stream ended without response.completed");
  if (!Array.isArray(completed.output) || completed.output.length === 0) {
    completed.output = [...outputItems.entries()].sort(([left], [right]) => left - right).map(([, item]) => item);
  }
  return completed;
}

export function openAiResponsesResponseShape(value: unknown): MessageResponse {
  if (!isRecord(value) || !Array.isArray(value.output)) {
    throw new Error("OpenAI Responses server returned an unexpected response shape");
  }
  const content: AnthropicContentBlock[] = [];
  for (const [index, rawItem] of value.output.entries()) {
    if (!isRecord(rawItem) || typeof rawItem.type !== "string") {
      throw new Error(`OpenAI Responses response has invalid output item ${index}`);
    }
    if (rawItem.type === "message") {
      const text = parseResponseText(rawItem.content);
      if (text) content.push({ type: "text", text });
      continue;
    }
    if (rawItem.type === "function_call") {
      const id = typeof rawItem.call_id === "string" && rawItem.call_id.length > 0
        ? rawItem.call_id
        : typeof rawItem.id === "string" && rawItem.id.length > 0
          ? rawItem.id
          : undefined;
      if (!id || typeof rawItem.name !== "string" || rawItem.name.length === 0) {
        throw new Error(`OpenAI Responses response has incomplete function call ${index}`);
      }
      let input: unknown = {};
      if (typeof rawItem.arguments === "string" && rawItem.arguments.trim()) {
        try {
          input = JSON.parse(rawItem.arguments);
        } catch (error) {
          throw new Error(`OpenAI Responses function call ${index} has invalid JSON arguments: ${error instanceof Error ? error.message : String(error)}`);
        }
      } else if (rawItem.arguments !== undefined) {
        input = rawItem.arguments;
      }
      content.push({ type: "tool_use", id, name: rawItem.name, input });
    }
  }
  const usage = responseUsage(value);
  const incomplete = isRecord(value.incomplete_details) && typeof value.incomplete_details.reason === "string"
    ? value.incomplete_details.reason
    : null;
  return {
    content,
    stop_reason: content.some((block) => block.type === "tool_use") ? "tool_calls" : incomplete ?? "stop",
    usage: {
      input_tokens: usage.input,
      output_tokens: usage.output,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: isRecord(value.usage) && isRecord(value.usage.input_tokens_details) && typeof value.usage.input_tokens_details.cached_tokens === "number"
        ? value.usage.input_tokens_details.cached_tokens
        : 0,
      thinking_tokens: usage.thinking,
    },
  };
}
