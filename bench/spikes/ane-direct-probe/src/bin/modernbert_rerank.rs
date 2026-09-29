//! Pair logits using the shared direct-ANE encoder and an fp32 CPU head.
#[allow(dead_code)]
#[path = "modernbert_full.rs"]
mod encoder;

fn main() -> anyhow::Result<()> {
    encoder::reranker::main()
}
