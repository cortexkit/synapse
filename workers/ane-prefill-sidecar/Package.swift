// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AnePrefillSidecar",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "AnePrefillSidecar", targets: ["AnePrefillSidecar"]),
        .executable(name: "ane-prefill-sidecar", targets: ["AnePrefillSidecarExecutable"]),
        .executable(name: "ane-prefill-sidecar-tests", targets: ["AnePrefillSidecarTests"]),
    ],
    targets: [
        .target(name: "AnePrefillSidecar"),
        .executableTarget(
            name: "AnePrefillSidecarExecutable",
            dependencies: ["AnePrefillSidecar"]
        ),
        .executableTarget(
            name: "AnePrefillSidecarTests",
            dependencies: ["AnePrefillSidecar"]
        ),
    ],
    swiftLanguageModes: [.v6]
)
