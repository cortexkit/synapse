import CoreML
import Darwin
import FluidAudio
import Foundation

struct Utterance: Decodable { let id: String; let path: String }
struct Request: Decodable { let utterances: [Utterance] }
struct Transcript: Encodable { let id: String; let text: String; let seconds: Double }
struct Output: Encodable {
    let engine = "FluidAudio"
    let version = "git-66a239a"
    let commit = "66a239a32609d3f90b50a9634e4b2820ab977011"
    let backend: String
    let cold_start_seconds: Double
    let warm_wall_seconds: Double
    let peak_rss_bytes: Int
    let gpu_peak_bytes: Int? = nil
    let energy_joules: Double? = nil
    let transcripts: [Transcript]
}

@main
struct Runner {
    static func argument(_ name: String) throws -> String {
        let args = CommandLine.arguments
        guard let index = args.firstIndex(of: name), index + 1 < args.count else {
            throw NSError(domain: "STTFluidAudio", code: 1, userInfo: [NSLocalizedDescriptionKey: "missing \(name)"])
        }
        return args[index + 1]
    }

    static func transcribe(_ utterance: Utterance, manager: AsrManager) async throws -> String {
        // Decoder state belongs to one utterance; carrying it between unrelated
        // corpus files would leak linguistic context into measured predictions.
        var state = try TdtDecoderState()
        return try await manager.transcribe(URL(fileURLWithPath: utterance.path), decoderState: &state).text
    }

    static func main() async throws {
        let backend = try argument("--backend")
        guard backend == "ane-pal6" || backend == "ane-int8" else {
            throw NSError(domain: "STTFluidAudio", code: 2, userInfo: [NSLocalizedDescriptionKey: "backend must be ane-pal6 or ane-int8"])
        }
        let request = try JSONDecoder().decode(Request.self, from: Data(contentsOf: URL(fileURLWithPath: argument("--request"))))
        let start = ProcessInfo.processInfo.systemUptime
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        let models = try AsrModels.loadLocal(
            from: URL(fileURLWithPath: argument("--model")), version: .v3,
            configuration: configuration, encoderPrecision: backend == "ane-pal6" ? .int8 : .int8V2,
            encoderComputeUnits: .cpuAndNeuralEngine)
        let manager = AsrManager()
        try await manager.loadModels(models)
        if let first = request.utterances.first { _ = try await transcribe(first, manager: manager) }
        let cold = ProcessInfo.processInfo.systemUptime - start
        let warm = ProcessInfo.processInfo.systemUptime
        var transcripts: [Transcript] = []
        for utterance in request.utterances {
            let before = ProcessInfo.processInfo.systemUptime
            let text = try await transcribe(utterance, manager: manager)
            transcripts.append(Transcript(id: utterance.id, text: text, seconds: ProcessInfo.processInfo.systemUptime - before))
        }
        var usage = rusage()
        guard getrusage(RUSAGE_SELF, &usage) == 0 else { throw POSIXError(.EIO) }
        let output = Output(backend: backend, cold_start_seconds: cold,
                            warm_wall_seconds: ProcessInfo.processInfo.systemUptime - warm,
                            peak_rss_bytes: usage.ru_maxrss, transcripts: transcripts)
        try JSONEncoder().encode(output).write(to: URL(fileURLWithPath: argument("--output")), options: .atomic)
    }
}
