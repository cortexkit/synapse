import CoreML
import Darwin
import FluidAudio
import Foundation

struct Clip: Decodable {
    let id: String
    let pcm: String
    let lists: [String: [String]]
}

struct Request: Decodable {
    let clips: [Clip]
    let tdt: String
    let ctc: String
    let output: String
}

struct Prediction: Encodable {
    let id: String
    let system = "fluidaudio-ctc-default"
    let list: String
    let text: String
    let seconds: Double
}

enum BenchError: Error { case invalidRequest, missingTimings, droppedTerms, memoryLimit }

@main
struct Runner {
    static func samples(_ clip: Clip) throws -> [Float] {
        let data = try Data(contentsOf: URL(fileURLWithPath: clip.pcm))
        guard !data.isEmpty, data.count % 4 == 0 else { throw BenchError.invalidRequest }
        return data.withUnsafeBytes { raw in
            stride(from: 0, to: data.count, by: 4).map { raw.loadUnaligned(fromByteOffset: $0, as: Float.self) }
        }
    }

    static func transcribe(_ audio: [Float], terms: [String], manager: AsrManager,
                           ctc: CtcModels) async throws -> String {
        var state = try TdtDecoderState()
        let result = try await manager.transcribe(audio, decoderState: &state)
        if terms.isEmpty { return result.text }
        guard let timings = result.tokenTimings, !timings.isEmpty else { throw BenchError.missingTimings }
        let vocabulary = CustomVocabularyContext(terms: terms.map { CustomVocabularyTerm(text: $0, weight: 10) })
        let session = try await VocabularyBoostingSession(vocabulary: vocabulary, ctcModels: ctc)
        guard session.vocabulary.terms.count == terms.count else { throw BenchError.droppedTerms }
        // The pinned runtime logs CTC failures and returns nil, just as for no correction.
        // The Python supervisor rejects such warnings rather than silently scoring a fallback.
        return await session.rescore(text: result.text, tokenTimings: timings, audioSamples: audio)?.text ?? result.text
    }

    static func main() async throws {
        AppLogger.minimumLevel = .warning
        AppLogger.mirrorsToConsole = true
        guard CommandLine.arguments.count == 2 else { throw BenchError.invalidRequest }
        let request = try JSONDecoder().decode(Request.self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
        guard !request.clips.isEmpty, !FileManager.default.fileExists(atPath: request.output) else {
            throw BenchError.invalidRequest
        }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let models = try AsrModels.loadLocal(from: URL(fileURLWithPath: request.tdt), version: .v3,
                                             configuration: config, encoderPrecision: .int8,
                                             encoderComputeUnits: .cpuAndNeuralEngine)
        let manager = AsrManager()
        try await manager.loadModels(models)
        let ctc = try await CtcModels.loadDirect(from: URL(fileURLWithPath: request.ctc), variant: .ctc110m)
        let first = request.clips[0]
        _ = try await transcribe(samples(first), terms: first.lists["local"] ?? [], manager: manager, ctc: ctc)
        guard FileManager.default.createFile(atPath: request.output, contents: nil,
                                             attributes: [.posixPermissions: 0o600]) else { throw BenchError.invalidRequest }
        let output = try FileHandle(forWritingTo: URL(fileURLWithPath: request.output))
        defer { try? output.close() }
        let encoder = JSONEncoder()
        var count = 0
        for clip in request.clips {
            for name in ["none", "local", "global", "unrelated"] {
                guard let terms = clip.lists[name] else { continue }
                let start = ProcessInfo.processInfo.systemUptime
                let text = try await transcribe(samples(clip), terms: terms, manager: manager, ctc: ctc)
                let seconds = ProcessInfo.processInfo.systemUptime - start
                var usage = rusage()
                guard getrusage(RUSAGE_SELF, &usage) == 0, usage.ru_maxrss < 20 * 1024 * 1024 * 1024 else {
                    throw BenchError.memoryLimit
                }
                let row = Prediction(id: clip.id, list: name, text: text, seconds: seconds)
                try output.write(contentsOf: encoder.encode(row) + Data([10]))
                count += 1
                if count % 50 == 0 { print("FluidAudio: \(count) predictions") }
            }
        }
        print("FluidAudio complete: \(count) predictions")
    }
}
