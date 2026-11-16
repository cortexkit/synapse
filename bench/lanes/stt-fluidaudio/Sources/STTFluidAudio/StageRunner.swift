#if STAGE_KIT
import CoreML
import Darwin
import FluidAudio
import Foundation

enum StageRunner {
    static func lockLane() throws -> Int32 {
        guard ProcessInfo.processInfo.environment["TMPDIR"] == nil else {
            throw NSError(domain: "StageKit: unset TMPDIR", code: 1)
        }
        let directory = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Caches/ck-synapse")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let descriptor = open(directory.appendingPathComponent("ane-direct.lock").path, O_RDWR | O_CREAT, 0o600)
        guard descriptor >= 0 else { throw POSIXError(.EIO) }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            close(descriptor)
            throw NSError(domain: "ane_lane_busy", code: 1)
        }
        return descriptor
    }

    static func units(_ model: MLModel) -> String {
        switch model.configuration.computeUnits {
        case .cpuOnly: return "cpuOnly"
        case .cpuAndGPU: return "cpuAndGPU"
        case .cpuAndNeuralEngine: return "cpuAndNeuralEngine"
        case .all: return "all (GPU permitted)"
        @unknown default: return "unknown-\(model.configuration.computeUnits.rawValue)"
        }
    }

    static func run(_ request: Request, manager: AsrManager, models: AsrModels, backend: String, cold: Double) async throws {
        guard backend == "ane-pal6", request.utterances.count == 1200,
              Set(request.utterances.map(\.id)).count == 1200 else {
            throw NSError(domain: "StageKit requires pinned pal6 assets and 1200 unique clips", code: 1)
        }
        let runID = try Runner.argument("--stage-run-id")
        let instrumented = ProcessInfo.processInfo.environment["ANE_PARAKEET_HOST_TIMING"] == "1"
        let outputURL = URL(fileURLWithPath: try Runner.argument("--stage-output"))
        FileManager.default.createFile(atPath: outputURL.path, contents: nil)
        let file = try FileHandle(forWritingTo: outputURL)
        defer { try? file.close() }
        var loaded = ["preprocessor": units(models.preprocessor), "predictor": units(models.decoder),
                      "joint": units(models.joint)]
        if let encoder = models.encoder { loaded["encoder"] = units(encoder) }
        for (index, function) in models.encoderLadder.enumerated() {
            loaded["encoder_ladder_\(index)"] = units(function.model)
        }
        let metadata: [String: Any] = ["source_revision": "66a239a32609d3f90b50a9634e4b2820ab977011",
            "asset_revision": "7dd20fe6b1797d35f5e3307e8b1732d9a178edfe", "loaded_compute_units": loaded,
            "placement_note": "allowed compute units, not observed device placement",
            "model_boundary": "sum of encoder-input preparation through raw TDT hypothesis per segment; excludes frontend and text formatting",
            "request_boundary": "original Runner.transcribe URL-to-text including fresh decoder state",
            "segment_note": "window/recovery model calls aggregated per clip; overlapping windows may execute concurrently",
            "warm_wall_note": "whole pass includes unmeasured same-clip warmups; cold_start is model loading only"]
        let warmStart = ProcessInfo.processInfo.systemUptime
        var transcripts: [[String: Any]] = []
        for utterance in request.utterances {
            // Prime the same input and buffers before both instrumented and baseline measurements.
            let warmupStart = ProcessInfo.processInfo.systemUptime
            _ = try await Runner.transcribe(utterance, manager: manager)
            let warmup: [String: Any] = ["schema": "matched-stages-v1", "engine": "fluidaudio", "run_id": runID,
                "clip_id": utterance.id, "invocation_id": "warmup", "population": "warmup", "warm": false,
                "observation": "off", "instrumented": false, "segment_id": 0,
                "request_total_ms": (ProcessInfo.processInfo.systemUptime - warmupStart) * 1000]
            try file.write(contentsOf: JSONSerialization.data(withJSONObject: warmup, options: [.sortedKeys]))
            try file.write(contentsOf: Data([10]))
            let collector = StageKit(enabled: instrumented)
            let before = ProcessInfo.processInfo.systemUptime
            let text = try await StageKit.$current.withValue(collector) {
                try await Runner.transcribe(utterance, manager: manager)
            }
            let requestMS = (ProcessInfo.processInfo.systemUptime - before) * 1000
            var record = collector.snapshot()
            guard record["model_total_ms"] != nil else {
                throw NSError(domain: "StageKit model hook not reached", code: 1)
            }
            record.merge(["schema": "matched-stages-v1", "engine": "fluidaudio", "run_id": runID,
                "clip_id": utterance.id, "invocation_id": "primary", "segment_id": 0,
                "population": "corpus", "warm": true, "observation": "off", "instrumented": instrumented,
                "aggregated_segments": true,
                "request_total_ms": requestMS, "metadata": metadata, "profile": "parakeet-v3-CoreML-pal6",
                "text": text]) { _, new in new }
            // No per-step writes occur; the entire clip has completed before serialization.
            try file.write(contentsOf: JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]))
            try file.write(contentsOf: Data([10]))
            transcripts.append(["id": utterance.id, "text": text, "seconds": requestMS / 1000])
        }
        var usage = rusage()
        guard getrusage(RUSAGE_SELF, &usage) == 0 else { throw POSIXError(.EIO) }
        try JSONSerialization.data(withJSONObject: ["engine": "FluidAudio", "backend": backend,
            "version": "git-66a239a+timing", "commit": "66a239a32609d3f90b50a9634e4b2820ab977011",
            "cold_start_seconds": cold, "warm_wall_seconds": ProcessInfo.processInfo.systemUptime - warmStart,
            "peak_rss_bytes": usage.ru_maxrss,
            "metadata": metadata, "transcripts": transcripts], options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: try Runner.argument("--output")), options: .atomic)
    }
}
#endif
