use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{ensure, Result};
use clap::Parser;
use synapse_bench::rig_protocol::{
    read_json_frame, write_json_frame, CandidateMetadata, CandidateRequest, CandidateResponse,
    PROTOCOL_VERSION,
};
use tokenizers::{EncodeInput, Tokenizer, TruncationParams};

#[allow(dead_code)]
#[derive(Parser)]
struct Args {
    #[arg(long)]
    serve_stdio: bool,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long)]
    device: String,
    #[arg(long)]
    dtype: String,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    cuda_graphs: bool,
    #[arg(long)]
    execution: String,
    #[arg(long)]
    max_length: usize,
    #[arg(long)]
    attention_units: usize,
    #[arg(long)]
    shapes: String,
    #[arg(long)]
    package_cache: Option<PathBuf>,
    #[arg(long)]
    model_label: Option<String>,
}

fn main() -> Result<()> {
    let started = Instant::now();
    let args = Args::parse();
    ensure!(
        args.serve_stdio,
        "fixture candidate only supports --serve-stdio"
    );
    let mut tokenizer = Tokenizer::from_file(&args.tokenizer)
        .map_err(|error| anyhow::anyhow!("tokenizer: {error}"))?;
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: args.max_length,
            ..Default::default()
        }))
        .map_err(|error| anyhow::anyhow!("truncation: {error}"))?;

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut input = BufReader::new(stdin.lock());
    let mut output = BufWriter::new(stdout.lock());
    write_json_frame(
        &mut output,
        &CandidateResponse::Ready {
            protocol_version: PROTOCOL_VERSION,
            metadata: CandidateMetadata {
                lane: "fixture-cpu".to_owned(),
                model: args
                    .model_label
                    .unwrap_or_else(|| "fixture-model".to_owned()),
                provider: args.device,
                dtype: args.dtype,
                execution: args.execution,
                notes: "deterministic integration fixture".to_owned(),
                package_cache_root: None,
                internal_load_s: started.elapsed().as_secs_f64(),
                eager_shape_preload: false,
            },
        },
    )?;

    loop {
        let request: CandidateRequest = read_json_frame(&mut input)?;
        let response = match request {
            CandidateRequest::PrepareShapes { .. } => CandidateResponse::Prepared {
                internal_wall_s: 0.0,
            },
            CandidateRequest::Embed { texts, .. } => {
                let request_started = Instant::now();
                let tokens = texts
                    .iter()
                    .map(|text| {
                        tokenizer
                            .encode(text.as_str(), true)
                            .map(|encoding| {
                                encoding
                                    .get_attention_mask()
                                    .iter()
                                    .map(|&mask| u64::from(mask != 0))
                                    .sum::<u64>()
                            })
                            .map_err(|error| anyhow::anyhow!("encode fixture text: {error}"))
                    })
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .sum();
                CandidateResponse::Embedding {
                    vectors: texts.iter().map(|text| fixture_vector(text)).collect(),
                    reported_real_tokens: tokens,
                    internal_infer_wall_s: request_started.elapsed().as_secs_f64(),
                }
            }
            CandidateRequest::Rerank { pairs, .. } => {
                let request_started = Instant::now();
                let tokens = pairs
                    .iter()
                    .map(|pair| {
                        tokenizer
                            .encode(
                                EncodeInput::Dual(
                                    pair.query.clone().into(),
                                    pair.document.clone().into(),
                                ),
                                true,
                            )
                            .map(|encoding| {
                                encoding
                                    .get_attention_mask()
                                    .iter()
                                    .map(|&mask| u64::from(mask != 0))
                                    .sum::<u64>()
                            })
                            .map_err(|error| anyhow::anyhow!("encode fixture pair: {error}"))
                    })
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .sum();
                CandidateResponse::Rerank {
                    scores: pairs
                        .iter()
                        .map(|pair| fixture_score(&pair.query, &pair.document))
                        .collect(),
                    reported_real_tokens: tokens,
                    internal_infer_wall_s: request_started.elapsed().as_secs_f64(),
                }
            }
            CandidateRequest::Shutdown => CandidateResponse::Shutdown,
        };
        let shutdown = matches!(response, CandidateResponse::Shutdown);
        write_json_frame(&mut output, &response)?;
        if shutdown {
            return Ok(());
        }
    }
}

fn fixture_vector(text: &str) -> Vec<f32> {
    let byte_sum = text.bytes().map(f32::from).sum::<f32>();
    let mut vector = vec![byte_sum + 1.0, text.len() as f32 + 1.0, 1.0];
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    for value in &mut vector {
        *value /= norm;
    }
    vector
}

fn fixture_score(query: &str, document: &str) -> f32 {
    query
        .bytes()
        .chain(document.bytes())
        .map(f32::from)
        .sum::<f32>()
}
