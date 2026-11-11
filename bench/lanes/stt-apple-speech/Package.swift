// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "STTAppleSpeech",
    platforms: [.macOS(.v26)],
    products: [.executable(name: "stt-apple-speech", targets: ["STTAppleSpeech"])],
    targets: [.executableTarget(name: "STTAppleSpeech")]
)
