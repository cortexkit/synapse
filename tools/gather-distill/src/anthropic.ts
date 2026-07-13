import {
  applyClaudeCodeHeaders,
  applyClaudeCodeMetadata,
  buildBillingHeaderValue,
  CLAUDE_CODE_ENTRYPOINT,
  CLAUDE_CODE_IDENTITY,
  orderClaudeCodeBody,
  resolveClaudeCodeIdentity,
  signRequestBody,
} from "@cortexkit/anthropic-auth-core";
import type { Credential } from "./auth.ts";
import type { AnthropicContentBlock, TrajectoryMessage } from "./types.ts";
import type { ToolDeclaration } from "./tools.ts";
import { isRecord } from "./utils.ts";

const ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages";

export interface MessageRequest {
  model: string;
  max_tokens: number;
  system: string;
  messages: TrajectoryMessage[];
  tools?: ToolDeclaration[];
  tool_choice?: { type: "none" };
  effort?: "low" | "medium" | "high";
}

export interface MessageResponse {
  content: AnthropicContentBlock[];
  stop_reason: string | null;
  usage: {
    input_tokens: number;
    output_tokens: number;
    cache_creation_input_tokens: number;
    cache_read_input_tokens: number;
  };
}

// 5-minute ephemeral cache. Our reuse is WITHIN a single trajectory (turns
// seconds apart, whole gather completes in well under 5 minutes); we never
// reuse a prefix across trajectories, so paying the 1h cache-write premium
// would be waste. Default ephemeral TTL is 5m, so no ttl field is set.
const EPHEMERAL_CACHE = { type: "ephemeral" } as const;

/**
 * Place 5m cache breakpoints on the stable prefix (tools + system) and on the
 * growing message prefix (last block of the last message). Anthropic matches
 * the longest cached prefix, so marking only the current tail each turn lets
 * the previous turn's accumulated tool results register as a cache read.
 * Mutates only freshly-cloned structures so the caller's persistent trajectory
 * is never touched (breakpoints must not accumulate across turns).
 */
function applyPromptCache(body: Record<string, unknown>): void {
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    const tools = body.tools as Record<string, unknown>[];
    const last = tools[tools.length - 1]!;
    body.tools = [...tools.slice(0, -1), { ...last, cache_control: EPHEMERAL_CACHE }];
  }

  if (Array.isArray(body.system) && body.system.length > 0) {
    const system = body.system as Record<string, unknown>[];
    const last = system[system.length - 1]!;
    body.system = [...system.slice(0, -1), { ...last, cache_control: EPHEMERAL_CACHE }];
  } else if (typeof body.system === "string") {
    body.system = [{ type: "text", text: body.system, cache_control: EPHEMERAL_CACHE }];
  }

  if (Array.isArray(body.messages) && body.messages.length > 0) {
    const messages = body.messages as Record<string, unknown>[];
    const lastIndex = messages.length - 1;
    const lastMessage = messages[lastIndex]!;
    const content = lastMessage.content;
    let cachedContent: unknown[];
    if (typeof content === "string") {
      cachedContent = [{ type: "text", text: content, cache_control: EPHEMERAL_CACHE }];
    } else if (Array.isArray(content) && content.length > 0) {
      const lastBlock = content[content.length - 1] as Record<string, unknown>;
      cachedContent = [...content.slice(0, -1), { ...lastBlock, cache_control: EPHEMERAL_CACHE }];
    } else {
      return;
    }
    messages[lastIndex] = { ...lastMessage, content: cachedContent };
  }
}

export class AccountRejectedError extends Error {
  readonly status: number;

  constructor(status: number) {
    super(status === 401 ? "Anthropic rejected the account credentials" : "Anthropic account is temporarily unavailable");
    this.name = "AccountRejectedError";
    this.status = status;
  }
}

function responseShape(value: unknown): MessageResponse {
  if (!isRecord(value) || !Array.isArray(value.content) || !isRecord(value.usage)) {
    throw new Error("Anthropic returned an unexpected response shape");
  }
  const input = value.usage.input_tokens;
  const output = value.usage.output_tokens;
  if (typeof input !== "number" || typeof output !== "number") {
    throw new Error("Anthropic response is missing token usage");
  }
  const cacheCreation = value.usage.cache_creation_input_tokens;
  const cacheRead = value.usage.cache_read_input_tokens;
  return {
    content: value.content as AnthropicContentBlock[],
    stop_reason: typeof value.stop_reason === "string" ? value.stop_reason : null,
    usage: {
      input_tokens: input,
      output_tokens: output,
      cache_creation_input_tokens: typeof cacheCreation === "number" ? cacheCreation : 0,
      cache_read_input_tokens: typeof cacheRead === "number" ? cacheRead : 0,
    },
  };
}

export async function sendMessage(credential: Credential, request: MessageRequest): Promise<MessageResponse> {
  const headers = new Headers({
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
  });
  const messages = request.messages.map(({ role, content }) => ({ role, content }));
  const body: Record<string, unknown> = {
    model: request.model,
    max_tokens: request.max_tokens,
    system: request.system,
    messages,
  };
  if (request.tools) body.tools = request.tools;
  if (request.tool_choice) body.tool_choice = request.tool_choice;
  if (request.effort) body.output_config = { effort: request.effort };

  let bodyText: string;
  if (credential.kind === "oauth") {
    const identity = await resolveClaudeCodeIdentity(credential.secret, request.model);
    const billingHeader = buildBillingHeaderValue(messages, undefined, CLAUDE_CODE_ENTRYPOINT);
    body.system = [
      { type: "text", text: billingHeader },
      { type: "text", text: CLAUDE_CODE_IDENTITY },
      { type: "text", text: request.system },
    ];
    applyPromptCache(body);
    applyClaudeCodeMetadata(body, identity);
    applyClaudeCodeHeaders(headers, credential.secret, { body, identity });
    bodyText = await signRequestBody(JSON.stringify(orderClaudeCodeBody(body)));
  } else {
    headers.set("x-api-key", credential.secret);
    applyPromptCache(body);
    bodyText = JSON.stringify(body);
  }

  const response = await fetch(ANTHROPIC_MESSAGES_URL, {
    method: "POST",
    headers,
    body: bodyText,
  });
  if (response.status === 401 || response.status === 429) throw new AccountRejectedError(response.status);
  if (!response.ok) {
    const requestId = response.headers.get("request-id");
    throw new Error(`Anthropic messages request failed (${response.status}${requestId ? `, request ${requestId}` : ""})`);
  }
  return responseShape(await response.json());
}

export function assistantText(content: AnthropicContentBlock[]): string {
  return content
    .filter((block): block is Extract<AnthropicContentBlock, { type: "text" }> => block.type === "text")
    .map((block) => block.text)
    .join("\n");
}
