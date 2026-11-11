// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "STTFluidAudio",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "stt-fluidaudio", targets: ["STTFluidAudio"])],
    dependencies: [.package(url: "https://github.com/FluidInference/FluidAudio.git",
                            revision: "66a239a32609d3f90b50a9634e4b2820ab977011")],
    targets: [.executableTarget(name: "STTFluidAudio",
                               dependencies: [.product(name: "FluidAudio", package: "FluidAudio")])]
)
