import fs from 'node:fs';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { finished } from 'node:stream/promises';
import { pipeline, AutoTokenizer } from '@huggingface/transformers';

const MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
const ORT_SNAPSHOT_DIR = path.join(
  os.homedir(),
  '.cache/huggingface/hub/models--Qdrant--all-MiniLM-L6-v2-onnx/snapshots/manual',
);
const ORT_MODEL_PATH = path.join(ORT_SNAPSHOT_DIR, 'model.onnx');
const MAX_LENGTH = 512;
const MAX_BATCH_ITEMS = 64;
const MAX_BATCH_TOKENS = 16_384;
const RSS_SAMPLE_MS = 50;

const HELP = `Usage:
  bun main.mjs --engine transformersjs --corpus <jsonl> --out <json> [--vectors-out <jsonl>] [--limit N] --model-label <label> [--dtype default|fp32] [--prefix-document <str>]
  node main.mjs --engine ort-node --corpus <jsonl> --out <json> [--vectors-out <jsonl>] [--limit N] --model-label <label> [--prefix-document <str>]

Required flags:
  --engine transformersjs|ort-node
  --corpus <jsonl>
  --out <json>
  --model-label <label>

Optional flags:
  --vectors-out <jsonl>
  --limit <n>
  --dtype <label>   Transformers.js only; "default" maps to the shipped q8 profile. Default: default
  --prefix-document <str>  Prepend this string to every corpus text before tokenization
`;

function parseArgs(argv) {
  const args = {
    dtype: 'default',
  };

  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      args.help = true;
      continue;
    }
    if (!token.startsWith('--')) {
      throw new Error(`Unexpected argument: ${token}`);
    }

    const key = token.slice(2);
    const value = argv[i + 1];
    if (value == null || value.startsWith('--')) {
      throw new Error(`Missing value for --${key}`);
    }
    args[key.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())] = value;
    i += 1;
  }

  if (args.help) {
    return args;
  }

  for (const required of ['engine', 'corpus', 'out', 'modelLabel']) {
    if (!args[required]) {
      throw new Error(`Missing required flag --${required.replace(/[A-Z]/g, (m) => `-${m.toLowerCase()}`)}`);
    }
  }

  if (!['transformersjs', 'ort-node'].includes(args.engine)) {
    throw new Error(`Unsupported --engine ${args.engine}`);
  }

  if (args.limit != null) {
    args.limit = Number.parseInt(args.limit, 10);
    if (!Number.isInteger(args.limit) || args.limit <= 0) {
      throw new Error('--limit must be a positive integer');
    }
  }

  if (args.engine === 'ort-node' && args.dtype !== 'default') {
    throw new Error('--dtype is only supported with --engine transformersjs');
  }

  return args;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    process.stdout.write(HELP);
    return;
  }

  let peakRss = process.memoryUsage().rss;
  const sampleRss = () => {
    peakRss = Math.max(peakRss, process.memoryUsage().rss);
  };
  const rssTimer = setInterval(sampleRss, RSS_SAMPLE_MS);
  rssTimer.unref?.();

  try {
    const result = args.engine === 'transformersjs'
      ? await runTransformersJsLane(args, sampleRss)
      : await runOrtNodeLane(args, sampleRss);

    sampleRss();
    result.self_peak_rss_bytes = peakRss;
    await writeJson(args.out, result);
  } finally {
    clearInterval(rssTimer);
  }
}

async function loadCorpus(corpusPath, limit, prefixDocument) {
  const rows = [];
  const input = await fsp.readFile(corpusPath, 'utf8');
  for (const [index, line] of input.split(/\r?\n/).entries()) {
    if (!line.trim()) {
      continue;
    }
    let row;
    try {
      row = JSON.parse(line);
    } catch (error) {
      throw new Error(`Invalid JSON on line ${index + 1} of ${corpusPath}: ${error.message}`);
    }
    if (typeof row.id !== 'string' || typeof row.text !== 'string') {
      throw new Error(`Corpus row ${index + 1} must contain string id and text fields`);
    }
    rows.push({ id: row.id, text: applyPrefix(prefixDocument, row.text) });
    if (limit != null && rows.length >= limit) {
      break;
    }
  }
  if (rows.length === 0) {
    throw new Error(`Corpus is empty: ${corpusPath}`);
  }
  return rows;
}

async function runTransformersJsLane(args, sampleRss) {
  const { pipelineDtype, noteDtype } = resolveTransformersDtype(args.dtype);
  const extractor = await pipeline(
    'feature-extraction',
    MODEL_ID,
    pipelineDtype ? { dtype: pipelineDtype } : {},
  );

  try {
    await extractor(applyPrefix(args.prefixDocument, 'warmup'), { pooling: 'mean', normalize: true });
    sampleRss();
    const coldLoadS = process.uptime();

    const corpus = await loadCorpus(args.corpus, args.limit, args.prefixDocument);
    const tokenized = pretokenizeCorpus(extractor.tokenizer, corpus);
    const batches = buildBatches(tokenized);
    const vectorsWriter = createVectorsWriter(args.vectorsOut);

    const inferStarted = performance.now();
    try {
      for (const batch of batches) {
        const embeddings = await extractor(batch.map((item) => item.text), {
          pooling: 'mean',
          normalize: true,
        });
        const vectors = embeddings.tolist();
        for (let i = 0; i < batch.length; i += 1) {
          writeVector(vectorsWriter, batch[i].id, vectors[i]);
        }
        sampleRss();
      }
    } finally {
      await closeVectorsWriter(vectorsWriter);
    }
    const inferWallS = (performance.now() - inferStarted) / 1000;

    return makeLaneResult({
      lane: 'ts-transformersjs',
      model: args.modelLabel,
      coldLoadS,
      inferWallS,
      inputTokens: sumInputTokens(tokenized),
      items: corpus.length,
      notes: [
        'engine=transformersjs',
        `dtype=${noteDtype}`,
        'pooling=mean',
        'normalize=l2',
        `batch=sum_tokens<=${MAX_BATCH_TOKENS},items<=${MAX_BATCH_ITEMS}`,
        `max_length=${MAX_LENGTH}`,
        `prefix_document=${formatPrefixNote(args.prefixDocument)}`,
      ].join(', '),
    });
  } finally {
    await extractor.dispose?.();
  }
}

// The benchmark's "default" Transformers.js run must match the shipped
// MiniLM production profile, which uses the quantized q8 weights on CPU.
function resolveTransformersDtype(dtypeLabel) {
  if (dtypeLabel === 'default') {
    return {
      pipelineDtype: 'q8',
      noteDtype: 'default(q8)',
    };
  }

  return {
    pipelineDtype: dtypeLabel,
    noteDtype: dtypeLabel,
  };
}

async function runOrtNodeLane(args, sampleRss) {
  ensureLocalOrtSnapshot();

  const tokenizer = await AutoTokenizer.from_pretrained(ORT_SNAPSHOT_DIR, { local_files_only: true });

  let ort;
  try {
    ort = await import('onnxruntime-node');
  } catch (error) {
    if (typeof Bun !== 'undefined') {
      throw new Error(`onnxruntime-node failed to load under bun: ${error.message}. Re-run with node main.mjs.`);
    }
    throw error;
  }
  const ortApi = ort.default ?? ort;
  const session = await ortApi.InferenceSession.create(ORT_MODEL_PATH);
  const inputNames = new Set(session.inputNames);

  await embedOrtBatch(ortApi, session, inputNames, tokenizer, [applyPrefix(args.prefixDocument, 'warmup')]);
  sampleRss();
  const coldLoadS = process.uptime();

  const corpus = await loadCorpus(args.corpus, args.limit, args.prefixDocument);
  const tokenized = pretokenizeCorpus(tokenizer, corpus);
  const batches = buildBatches(tokenized);
  const vectorsWriter = createVectorsWriter(args.vectorsOut);

  const inferStarted = performance.now();
  try {
    for (const batch of batches) {
      const vectors = await embedOrtBatch(
        ortApi,
        session,
        inputNames,
        tokenizer,
        batch.map((item) => item.text),
      );
      for (let i = 0; i < batch.length; i += 1) {
        writeVector(vectorsWriter, batch[i].id, vectors[i]);
      }
      sampleRss();
    }
  } finally {
    await closeVectorsWriter(vectorsWriter);
  }
  const inferWallS = (performance.now() - inferStarted) / 1000;

  return makeLaneResult({
    lane: 'ts-ort-node',
    model: args.modelLabel,
    coldLoadS,
    inferWallS,
    inputTokens: sumInputTokens(tokenized),
    items: corpus.length,
    notes: [
      'engine=ort-node',
      'dtype=fp32',
      `model=${ORT_MODEL_PATH}`,
      `tokenizer=${path.join(ORT_SNAPSHOT_DIR, 'tokenizer.json')}`,
      'pooling=mean',
      'normalize=l2',
      `batch=sum_tokens<=${MAX_BATCH_TOKENS},items<=${MAX_BATCH_ITEMS}`,
      `max_length=${MAX_LENGTH}`,
      `prefix_document=${formatPrefixNote(args.prefixDocument)}`,
    ].join(', '),
  });
}

function applyPrefix(prefixDocument, text) {
  return prefixDocument == null ? text : `${prefixDocument}${text}`;
}

function formatPrefixNote(prefixDocument) {
  return prefixDocument == null ? 'none' : JSON.stringify(prefixDocument);
}

function pretokenizeCorpus(tokenizer, corpus) {
  return corpus.map(({ id, text }) => {
    const encoded = tokenizer(text, {
      truncation: true,
      max_length: MAX_LENGTH,
      return_tensor: false,
    });
    return {
      id,
      text,
      inputTokens: encoded.input_ids.length,
    };
  });
}

function buildBatches(items) {
  const batches = [];
  let start = 0;
  while (start < items.length) {
    let end = start;
    let tokenSum = 0;
    while (end < items.length && end - start < MAX_BATCH_ITEMS) {
      const nextTokens = items[end].inputTokens;
      if (end > start && tokenSum + nextTokens > MAX_BATCH_TOKENS) {
        break;
      }
      tokenSum += nextTokens;
      end += 1;
    }
    batches.push(items.slice(start, end));
    start = end;
  }
  return batches;
}

async function embedOrtBatch(ort, session, inputNames, tokenizer, texts) {
  const encoded = tokenizer(texts, {
    padding: true,
    truncation: true,
    max_length: MAX_LENGTH,
    return_tensor: false,
  });
  const batchSize = encoded.input_ids.length;
  const seqLen = encoded.input_ids[0]?.length ?? 0;
  const feeds = {
    input_ids: new ort.Tensor('int64', toBigInt64Flat(encoded.input_ids), [batchSize, seqLen]),
    attention_mask: new ort.Tensor('int64', toBigInt64Flat(encoded.attention_mask), [batchSize, seqLen]),
  };
  if (inputNames.has('token_type_ids')) {
    feeds.token_type_ids = new ort.Tensor(
      'int64',
      toBigInt64Flat(encoded.token_type_ids ?? makeZeroMatrix(batchSize, seqLen)),
      [batchSize, seqLen],
    );
  }

  const outputs = await session.run(feeds);
  const hidden = outputs[session.outputNames[0]];
  return meanPoolAndNormalize(hidden.data, hidden.dims, encoded.attention_mask);
}

function toBigInt64Flat(rows) {
  const flat = new BigInt64Array(rows.length * (rows[0]?.length ?? 0));
  let offset = 0;
  for (const row of rows) {
    for (const value of row) {
      flat[offset] = BigInt(value);
      offset += 1;
    }
  }
  return flat;
}

function meanPoolAndNormalize(data, dims, attentionMask) {
  const [batchSize, seqLen, hiddenSize] = dims;
  const vectors = new Array(batchSize);
  for (let batch = 0; batch < batchSize; batch += 1) {
    const pooled = new Float32Array(hiddenSize);
    let count = 0;
    for (let token = 0; token < seqLen; token += 1) {
      if (attentionMask[batch][token] !== 1) {
        continue;
      }
      count += 1;
      const base = (batch * seqLen + token) * hiddenSize;
      for (let index = 0; index < hiddenSize; index += 1) {
        pooled[index] += data[base + index];
      }
    }
    const denom = count || 1;
    let normSquared = 0;
    for (let index = 0; index < hiddenSize; index += 1) {
      pooled[index] /= denom;
      normSquared += pooled[index] * pooled[index];
    }
    const norm = Math.sqrt(normSquared) || 1;
    const vector = new Array(hiddenSize);
    for (let index = 0; index < hiddenSize; index += 1) {
      vector[index] = pooled[index] / norm;
    }
    vectors[batch] = vector;
  }
  return vectors;
}

function sumInputTokens(items) {
  return items.reduce((sum, item) => sum + item.inputTokens, 0);
}

function makeLaneResult({ lane, model, coldLoadS, inferWallS, inputTokens, items, notes }) {
  return {
    lane,
    workload: 'embed-corpus-v1',
    model,
    cold_load_s: coldLoadS,
    infer_wall_s: inferWallS,
    input_tokens: inputTokens,
    tok_per_s: inferWallS > 0 ? inputTokens / inferWallS : 0,
    items,
    parity_mean_cosine: null,
    self_peak_rss_bytes: null,
    notes,
  };
}

async function writeJson(filePath, value) {
  await fsp.mkdir(path.dirname(filePath), { recursive: true });
  await fsp.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function createVectorsWriter(filePath) {
  if (!filePath) {
    return null;
  }
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  return fs.createWriteStream(filePath, { encoding: 'utf8' });
}

function writeVector(writer, id, vec) {
  if (!writer) {
    return;
  }
  writer.write(`${JSON.stringify({ id, vec })}\n`);
}

async function closeVectorsWriter(writer) {
  if (!writer) {
    return;
  }
  writer.end();
  await finished(writer);
}

function ensureLocalOrtSnapshot() {
  if (!fs.existsSync(ORT_MODEL_PATH)) {
    throw new Error(`Local ORT model not found: ${ORT_MODEL_PATH}`);
  }
  const tokenizerPath = path.join(ORT_SNAPSHOT_DIR, 'tokenizer.json');
  if (!fs.existsSync(tokenizerPath)) {
    throw new Error(`Local tokenizer not found: ${tokenizerPath}`);
  }
}

function makeZeroMatrix(rows, cols) {
  return Array.from({ length: rows }, () => Array(cols).fill(0));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});
