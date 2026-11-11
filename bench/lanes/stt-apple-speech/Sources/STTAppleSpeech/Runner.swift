import AVFoundation
import Darwin
import Foundation
import Speech

struct Utterance: Decodable {
    let id: String
    let path: String
    let locale: String
}
struct Request: Decodable { let utterances: [Utterance] }
struct Transcript: Encodable {
    let id: String
    var text: String? = nil
    var seconds: Double? = nil
    var error: String? = nil
}
struct Output: Encodable {
    let engine = "Apple SpeechTranscriber"
    let version = ProcessInfo.processInfo.operatingSystemVersionString
    let commit = "Apple proprietary; source commit unavailable"
    let backend = "apple-speech"
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
            throw NSError(domain: "STTAppleSpeech", code: 1, userInfo: [NSLocalizedDescriptionKey: "missing \(name)"])
        }
        return args[index + 1]
    }

    static func transcribe(_ utterance: Utterance, locale: Locale) async throws -> String {
        let transcriber = SpeechTranscriber(
            locale: locale, transcriptionOptions: [], reportingOptions: [], attributeOptions: [])
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        let reader = Task { () throws -> String in
            var text = ""
            for try await result in transcriber.results where result.isFinal {
                let segment = String(result.text.characters)
                if !text.isEmpty && !segment.hasPrefix(" ") { text += " " }
                text += segment
            }
            return text
        }
        do {
            let file = try AVAudioFile(forReading: URL(fileURLWithPath: utterance.path))
            if let end = try await analyzer.analyzeSequence(from: file) {
                try await analyzer.finalizeAndFinish(through: end)
            } else {
                await analyzer.cancelAndFinishNow()
            }
            return try await reader.value
        } catch {
            reader.cancel()
            await analyzer.cancelAndFinishNow()
            throw error
        }
    }

    static func main() async throws {
        guard try argument("--backend") == "apple-speech" else {
            throw NSError(domain: "STTAppleSpeech", code: 2, userInfo: [NSLocalizedDescriptionKey: "backend must be apple-speech"])
        }
        let request = try JSONDecoder().decode(Request.self, from: Data(contentsOf: URL(fileURLWithPath: argument("--request"))))
        let start = ProcessInfo.processInfo.systemUptime
        var locales: [String: Locale] = [:]
        var unavailable: [String: String] = [:]
        for identifier in Set(request.utterances.map(\.locale)).sorted() {
            guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: Locale(identifier: identifier)) else {
                unavailable[identifier] = "SpeechTranscriber does not support \(identifier) on this OS"
                continue
            }
            let module = SpeechTranscriber(locale: locale, transcriptionOptions: [], reportingOptions: [], attributeOptions: [])
            let status = await AssetInventory.status(forModules: [module])
            if status != .installed {
                if CommandLine.arguments.contains("--install-assets") || CommandLine.arguments.contains("--install-assets-only") {
                    if let installation = try await AssetInventory.assetInstallationRequest(supporting: [module]) {
                        try await installation.downloadAndInstall()
                    }
                } else {
                    unavailable[identifier] = "Speech asset not installed; rerun worker with --install-assets explicitly"
                    continue
                }
            }
            locales[identifier] = locale
            if !CommandLine.arguments.contains("--install-assets-only"), let first = request.utterances.first(where: { $0.locale == identifier }) {
                _ = try await transcribe(first, locale: locale)
            }
        }
        if CommandLine.arguments.contains("--install-assets-only") {
            print("Installed locales: \(locales.keys.sorted()); unavailable: \(unavailable)")
            return
        }
        let cold = ProcessInfo.processInfo.systemUptime - start
        let warm = ProcessInfo.processInfo.systemUptime
        var transcripts: [Transcript] = []
        for utterance in request.utterances {
            guard let locale = locales[utterance.locale] else {
                transcripts.append(Transcript(id: utterance.id, error: unavailable[utterance.locale] ?? "locale unavailable"))
                continue
            }
            let before = ProcessInfo.processInfo.systemUptime
            do {
                let text = try await transcribe(utterance, locale: locale)
                transcripts.append(Transcript(id: utterance.id, text: text, seconds: ProcessInfo.processInfo.systemUptime - before))
            } catch {
                transcripts.append(Transcript(id: utterance.id, error: String(describing: error)))
            }
        }
        var usage = rusage()
        guard getrusage(RUSAGE_SELF, &usage) == 0 else { throw POSIXError(.EIO) }
        let output = Output(cold_start_seconds: cold, warm_wall_seconds: ProcessInfo.processInfo.systemUptime - warm,
                            peak_rss_bytes: usage.ru_maxrss, transcripts: transcripts)
        try JSONEncoder().encode(output).write(to: URL(fileURLWithPath: argument("--output")), options: .atomic)
    }
}
