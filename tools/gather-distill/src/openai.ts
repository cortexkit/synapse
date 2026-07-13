import type { MessageRequest, MessageResponse } from "./anthropic.ts";
import type { AnthropicContentBlock, TrajectoryMessage } from "./types.ts";
import type { ToolDeclaration } from "./tools.ts";
import { isRecord, sleep } from "./utils.ts";

const DEFAULT_OPENAI_BASE_URL = "http://127.0.0.1:8080/v1";
const DEFAULT_REQUEST_TIMEOUT_MS = 300_000;
const DEFAULT_TRANSIENT_RETRIES = 2;

export interface OpenAiToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

export interface OpenAiChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
  tool_calls?: OpenAiToolCall[];
  tool_call_id?: string;
}

export interface OpenAiToolDefinition {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

export interface OpenAiRequestOptions {
  baseUrl?: string;
  requestTimeoutMs?: number;
  transientRetries?: number;
}

export class OpenAiRequestTimeoutError extends Error {
  constructor(timeoutMs: number) {
    super(`OpenAI-compatible request timed out after ${timeoutMs}ms`);
    this.name = "OpenAiRequestTimeoutError";
  }
}

class OpenAiHttpError extends Error {
  readonly status: number;

  constructor(status: number, body: string) {
    super(`OpenAI-compatible chat completion failed (${status})${body ? `: ${body.slice(0, 1_000)}` : ""}`);
    this.name = "OpenAiHttpError";
    this.status = status;
  }
}

/** Convert the canonical Anthropic-shaped trajectory into chat.completions messages. */
export function toOpenAiMessages(system: string, messages: TrajectoryMessage[]): OpenAiChatMessage[] {
  const translated: OpenAiChatMessage[] = [{ role: "system", content: system }];
  for (const message of messages) translated.push(...translateTrajectoryMessage(message));
  return translated;
}

function toolArguments(input: unknown): string {
  return JSON.stringify(input ?? {}) ?? "{}";
}

function translateTrajectoryMessage(message: TrajectoryMessage): OpenAiChatMessage[] {
  if (typeof message.content === "string") {
    return [{ role: message.role, content: message.content }];
  }

  if (message.role === "assistant") {
    const text = message.content
      .filter((block): block is Extract<AnthropicContentBlock, { type: "text" }> => block.type === "text")
      .map((block) => block.text)
      .join("\n");
    const toolCalls = message.content
      .filter((block): block is Extract<AnthropicContentBlock, { type: "tool_use" }> => block.type === "tool_use")
      .map((block) => ({
        id: block.id,
        type: "function" as const,
        function: { name: block.name, arguments: toolArguments(block.input) },
      }));
    return [{ role: "assistant", content: text || null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) }];
  }

  const text = message.content
    .filter((block): block is Extract<AnthropicContentBlock, { type: "text" }> => block.type === "text")
    .map((block) => block.text)
    .join("\n");
  const toolResults = message.content.filter(
    (block): block is Extract<AnthropicContentBlock, { type: "tool_result" }> => block.type === "tool_result",
  );
  const translated: OpenAiChatMessage[] = text ? [{ role: "user", content: text }] : [];
  translated.push(
    ...toolResults.map((block) => ({
      role: "tool" as const,
      content: block.content,
      tool_call_id: block.tool_use_id,
    })),
  );
  return translated.length > 0 ? translated : [{ role: "user", content: "" }];
}

/** Render the unchanged AFT JSON schemas as OpenAI function definitions. */
export function toOpenAiTools(tools: ToolDeclaration[]): OpenAiToolDefinition[] {
  return tools.map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.input_schema,
    },
  }));
}

/**
 * Remove visible reasoning spans before final JSON extraction. llama-server
 * normally reports only total completion tokens, so the fallback count is a
 * whitespace-token estimate when no explicit reasoning-token field is present.
 */
export function stripThinkBlocks(text: string): { text: string; thinkingTokens: number } {
  let thinkingTokens = 0;
  const stripped = text.replace(/<think(?:\s[^>]*)?>([\s\S]*?)(?:<\/think>|$)/gi, (_match, reasoning: string) => {
    thinkingTokens += estimateWhitespaceTokens(reasoning);
    return "";
  });
  return { text: stripped, thinkingTokens };
}

function estimateWhitespaceTokens(text: string): number {
  const trimmed = text.trim();
  return trimmed.length === 0 ? 0 : trimmed.split(/\s+/u).length;
}

function contentText(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    return value
      .flatMap((part) => (isRecord(part) && typeof part.text === "string" ? [part.text] : []))
      .join("\n");
  }
  throw new Error("OpenAI-compatible response has non-text message content");
}

function parseToolArguments(value: unknown, index: number): unknown {
  if (value === undefined || (typeof value === "string" && value.trim().length === 0)) return {};
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch (error) {
    throw new Error(`OpenAI-compatible tool call ${index} has invalid JSON arguments: ${error instanceof Error ? error.message : String(error)}`);
  }
}

/** Translate an OpenAI assistant message into the trajectory's canonical blocks. */
export function fromOpenAiMessage(value: unknown): { content: AnthropicContentBlock[]; thinkingTokens: number } {
  if (!isRecord(value)) throw new Error("OpenAI-compatible response is missing a message object");
  const stripped = stripThinkBlocks(contentText(value.content));
  const content: AnthropicContentBlock[] = stripped.text.length > 0 ? [{ type: "text", text: stripped.text }] : [];
  const separateReasoning = [value.reasoning_content, value.reasoning, value.reasoning_text]
    .filter((reasoning): reasoning is string => typeof reasoning === "string")
    .join("\n");
  let thinkingTokens = stripped.thinkingTokens + estimateWhitespaceTokens(separateReasoning);

  if (value.tool_calls === undefined || value.tool_calls === null) return { content, thinkingTokens };
  if (!Array.isArray(value.tool_calls)) throw new Error("OpenAI-compatible response has invalid tool_calls");
  for (const [index, rawCall] of value.tool_calls.entries()) {
    if (!isRecord(rawCall) || !isRecord(rawCall.function)) {
      throw new Error(`OpenAI-compatible response has invalid tool call ${index}`);
    }
    const id = rawCall.id;
    const name = rawCall.function.name;
    if (typeof id !== "string" || id.length === 0 || typeof name !== "string" || name.length === 0) {
      throw new Error(`OpenAI-compatible response has incomplete tool call ${index}`);
    }
    content.push({
      type: "tool_use",
      id,
      name,
      input: parseToolArguments(rawCall.function.arguments, index),
    });
  }
  return { content, thinkingTokens };
}

function explicitThinkingTokens(usage: Record<string, unknown>): number | undefined {
  const direct = usage.reasoning_tokens ?? usage.thinking_tokens;
  if (typeof direct === "number" && Number.isFinite(direct) && direct >= 0) return direct;
  const details = usage.completion_tokens_details;
  if (isRecord(details)) {
    const reasoning = details.reasoning_tokens ?? details.thinking_tokens;
    if (typeof reasoning === "number" && Number.isFinite(reasoning) && reasoning >= 0) return reasoning;
  }
  return undefined;
}

/** Convert an OpenAI chat.completions payload into the existing message response shape. */
export function openAiResponseShape(value: unknown): MessageResponse {
  if (!isRecord(value) || !Array.isArray(value.choices) || !isRecord(value.usage)) {
    throw new Error("OpenAI-compatible server returned an unexpected response shape");
  }
  const choice = value.choices[0];
  if (!isRecord(choice) || !isRecord(choice.message)) {
    throw new Error("OpenAI-compatible response is missing choices[0].message");
  }
  const promptTokens = value.usage.prompt_tokens;
  const completionTokens = value.usage.completion_tokens;
  if (typeof promptTokens !== "number" || typeof completionTokens !== "number") {
    throw new Error("OpenAI-compatible response is missing token usage");
  }
  const translated = fromOpenAiMessage(choice.message);
  return {
    content: translated.content,
    stop_reason: typeof choice.finish_reason === "string" ? choice.finish_reason : null,
    usage: {
      input_tokens: promptTokens,
      output_tokens: completionTokens,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
      thinking_tokens: explicitThinkingTokens(value.usage) ?? translated.thinkingTokens,
    },
  };
}

export function openAiChatCompletionsUrl(baseUrl = DEFAULT_OPENAI_BASE_URL): string {
  const normalized = baseUrl.replace(/\/+$/, "");
  if (normalized.length === 0) throw new Error("OpenAI-compatible base URL must not be empty");
  return normalized.endsWith("/chat/completions") ? normalized : `${normalized}/chat/completions`;
}

export function isTransientOpenAiStatus(status: number): boolean {
  return status === 408 || status === 409 || status === 425 || status === 429 || status >= 500;
}

function isTransientOpenAiError(error: unknown): boolean {
  if (error instanceof OpenAiRequestTimeoutError) return true;
  if (error instanceof OpenAiHttpError) return isTransientOpenAiStatus(error.status);
  return error instanceof TypeError || (error instanceof Error && error.name === "AbortError");
}

async function fetchWithTimeout(url: string, body: string, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body,
      signal: controller.signal,
    });
  } catch (error) {
    if (controller.signal.aborted) throw new OpenAiRequestTimeoutError(timeoutMs);
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Send one local OpenAI-compatible chat completion. This lane intentionally
 * sends no credentials, prompt-cache controls, or Claude Code metadata.
 */
export async function sendOpenAiMessage(request: MessageRequest, options: OpenAiRequestOptions = {}): Promise<MessageResponse> {
  const timeoutMs = options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;
  const retries = options.transientRetries ?? DEFAULT_TRANSIENT_RETRIES;
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error("OpenAI-compatible request timeout must be positive");
  if (!Number.isInteger(retries) || retries < 0) throw new Error("OpenAI-compatible transient retries must be a non-negative integer");

  const body: Record<string, unknown> = {
    model: request.model,
    max_tokens: request.max_tokens,
    messages: toOpenAiMessages(request.system, request.messages),
  };
  if (request.tools !== undefined) body.tools = toOpenAiTools(request.tools);
  if (request.tool_choice) body.tool_choice = "none";
  const bodyText = JSON.stringify(body);
  const url = openAiChatCompletionsUrl(options.baseUrl);

  for (let attempt = 0; ; attempt += 1) {
    try {
      const response = await fetchWithTimeout(url, bodyText, timeoutMs);
      if (!response.ok) throw new OpenAiHttpError(response.status, await response.text());
      return openAiResponseShape(await response.json());
    } catch (error) {
      if (!isTransientOpenAiError(error) || attempt >= retries) throw error;
      await sleep(250 * 2 ** attempt);
    }
  }
}
