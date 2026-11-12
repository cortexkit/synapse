// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "STTBiasFluid",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "stt-bias-fluid", targets: ["STTBiasFluid"])],
    dependencies: [.package(url: "https://github.com/FluidInference/FluidAudio.git",
                            revision: "66a239a32609d3f90b50a9634e4b2820ab977011")],
    targets: [.executableTarget(name: "STTBiasFluid",
                               dependencies: [.product(name: "FluidAudio", package: "FluidAudio")])]
)
