import type { AftProcess, AftProcessFactory } from "../../src/tools.ts";

export interface RecordedRequest {
  id: string;
  command: string;
  name?: string;
  arguments?: Record<string, unknown>;
  [key: string]: unknown;
}

export type RequestHandler = (request: RecordedRequest, process: RecordedAftProcess) => void;

export class RecordedAftProcess implements AftProcess {
  readonly requests: RecordedRequest[] = [];
  readonly stdout: ReadableStream<Uint8Array>;
  readonly exited: Promise<number>;
  readonly stdin = {
    write: (line: string): void => {
      const request = JSON.parse(line) as RecordedRequest;
      this.requests.push(request);
      this.handler(request, this);
    },
  };

  private readonly encoder = new TextEncoder();
  private readonly resolveExit: (code: number) => void;
  private controller: ReadableStreamDefaultController<Uint8Array> | undefined;
  private exitedAlready = false;

  constructor(private readonly handler: RequestHandler) {
    this.stdout = new ReadableStream<Uint8Array>({
      start: (controller) => {
        this.controller = controller;
      },
    });
    let resolveExit!: (code: number) => void;
    this.exited = new Promise<number>((resolve) => {
      resolveExit = resolve;
    });
    this.resolveExit = resolveExit;
  }

  respond(value: Record<string, unknown>): void {
    this.controller?.enqueue(this.encoder.encode(`${JSON.stringify(value)}\n`));
  }

  exit(code = 1): void {
    if (this.exitedAlready) return;
    this.exitedAlready = true;
    this.controller?.close();
    this.resolveExit(code);
  }

  kill(): void {
    this.exit(137);
  }
}

export function recordedProcessFactory(...handlers: RequestHandler[]): {
  factory: AftProcessFactory;
  processes: RecordedAftProcess[];
} {
  const processes: RecordedAftProcess[] = [];
  return {
    factory: () => {
      const handler = handlers.shift();
      if (!handler) throw new Error("unexpected AFT process spawn");
      const process = new RecordedAftProcess(handler);
      processes.push(process);
      return process;
    },
    processes,
  };
}
