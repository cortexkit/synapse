//! Run CPU work while the hardware test repeats the same speech encoder input.
use anyhow::{ensure, Context, Result};
use serde_json::json;
use std::{
    hint::black_box,
    sync::{
        atomic::{AtomicBool, Ordering::Relaxed},
        mpsc, Arc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const MEMORY_THREADS: &str = "SYNAPSE_VULKAN_SPEECH_HOST_LOAD_MEMORY_THREADS";
const SPIN_THREADS: &str = "SYNAPSE_VULKAN_SPEECH_HOST_LOAD_SPIN_THREADS";
const MIB_PER_THREAD: &str = "SYNAPSE_VULKAN_SPEECH_HOST_LOAD_MIB";
const MIB: usize = 1024 * 1024;
const CHUNK_WORDS: usize = MIB / size_of::<u64>();
const SPIN_STEPS: u64 = 4096;

#[derive(Clone, Copy, Default)]
pub(super) struct Config {
    memory_threads: usize,
    spin_threads: usize,
    mib_per_thread: usize,
}

impl Config {
    pub(super) fn from_env() -> Option<Self> {
        let read = |name| {
            super::env(name).map(|value| {
                value
                    .parse::<usize>()
                    .unwrap_or_else(|_| panic!("{name} must be a nonnegative integer"))
            })
        };
        let memory_threads = read(MEMORY_THREADS);
        let spin_threads = read(SPIN_THREADS);
        let mib_per_thread = read(MIB_PER_THREAD);
        (memory_threads.is_some() || spin_threads.is_some() || mib_per_thread.is_some()).then_some(
            Self {
                memory_threads: memory_threads.unwrap_or(0),
                spin_threads: spin_threads.unwrap_or(0),
                mib_per_thread: mib_per_thread.unwrap_or(128),
            },
        )
    }

    fn validate(self) -> Result<()> {
        ensure!(
            self.memory_threads.saturating_add(self.spin_threads) <= 32,
            "host load supports at most 32 threads"
        );
        if self.memory_threads != 0 {
            ensure!(
                self.mib_per_thread > 0
                    && self.memory_threads.saturating_mul(self.mib_per_thread) <= 4096,
                "host load requires a positive buffer size and at most 4096 MiB total"
            );
        }
        Ok(())
    }
}

#[derive(Debug)]
enum Work {
    MemoryMiB(u64),
    SpinSteps(u64),
}

pub(super) struct Running {
    stop: Arc<AtomicBool>,
    threads: Vec<JoinHandle<Work>>,
    started: Instant,
}

impl Running {
    pub(super) fn start(config: Config) -> Result<Option<Self>> {
        config.validate()?;
        let n = config.memory_threads + config.spin_threads;
        if n == 0 {
            return Ok(None);
        }
        let mut load = Self {
            stop: Arc::new(AtomicBool::new(false)),
            threads: Vec::with_capacity(n),
            started: Instant::now(),
        };
        let (ready, readiness) = mpsc::channel();
        for i in 0..n {
            let stop = Arc::clone(&load.stop);
            let ready = ready.clone();
            let memory = i < config.memory_threads;
            let thread = thread::Builder::new()
                .name(format!("speech-host-load-{i}"))
                .spawn(move || {
                    if memory {
                        stream(config.mib_per_thread, &stop, ready)
                    } else {
                        spin(&stop, ready)
                    }
                })?;
            load.threads.push(thread);
        }
        drop(ready);
        for _ in 0..n {
            readiness
                .recv_timeout(Duration::from_secs(30))
                .context("host load workers did not start")?;
        }
        println!(
            "{}",
            json!({"vulkan_speech_host_load": {
                "memory_threads": config.memory_threads,
                "mib_per_memory_thread": config.mib_per_thread,
                "spin_threads": config.spin_threads,
            }})
        );
        Ok(Some(load))
    }

    fn stop_and_join(&mut self) -> impl Iterator<Item = thread::Result<Work>> + '_ {
        self.stop.store(true, Relaxed);
        self.threads.drain(..).map(JoinHandle::join)
    }

    pub(super) fn report(mut self) -> Result<()> {
        let elapsed = self.started.elapsed().as_secs_f64();
        let mut memory_mib = 0;
        let mut spin_steps = 0;
        let mut panicked = false;
        for work in self.stop_and_join() {
            match work {
                Ok(Work::MemoryMiB(mib)) => memory_mib += mib,
                Ok(Work::SpinSteps(steps)) => spin_steps += steps,
                Err(_) => panicked = true,
            }
        }
        ensure!(!panicked, "host load worker panicked");
        println!(
            "{}",
            json!({"vulkan_speech_host_load_done": {
                "memory_mib_touched": memory_mib,
                "spin_steps": spin_steps,
                "elapsed_s": elapsed,
            }})
        );
        Ok(())
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        for _ in self.stop_and_join() {}
    }
}

fn stream(mib: usize, stop: &AtomicBool, ready: mpsc::Sender<()>) -> Work {
    let mut words = vec![1u64; mib * CHUNK_WORDS];
    let mut ready = Some(ready);
    let mut touched = 0;
    loop {
        for chunk in words.as_chunks_mut::<CHUNK_WORDS>().0.iter_mut() {
            if stop.load(Relaxed) {
                return Work::MemoryMiB(touched);
            }
            for word in black_box(&mut *chunk) {
                *word = word.wrapping_add(1);
            }
            black_box(&*chunk);
            touched += 1;
            if let Some(ready) = ready.take() {
                let _ = ready.send(());
            }
        }
    }
}

fn spin(stop: &AtomicBool, ready: mpsc::Sender<()>) -> Work {
    let mut ready = Some(ready);
    let mut value = black_box(1u64);
    let mut steps = 0;
    while !stop.load(Relaxed) {
        for _ in 0..SPIN_STEPS {
            value ^= value << 13;
            value ^= value >> 7;
            value ^= value << 17;
        }
        value = black_box(value);
        steps += SPIN_STEPS;
        if let Some(ready) = ready.take() {
            let _ = ready.send(());
        }
    }
    Work::SpinSteps(steps)
}

#[test]
fn speech_repeat_host_load_starts_both_workers_and_joins_them() {
    let config = Config {
        memory_threads: 1,
        spin_threads: 1,
        mib_per_thread: 1,
    };
    let mut load = Running::start(config).unwrap().unwrap();
    let work: Vec<_> = load.stop_and_join().map(Result::unwrap).collect();
    assert!(
        matches!(&work[..], [Work::MemoryMiB(mib), Work::SpinSteps(steps)] if *mib > 0 && *steps >= SPIN_STEPS)
    );
    assert!(load.threads.is_empty());
    assert!(Running::start(Config::default()).unwrap().is_none());
}
