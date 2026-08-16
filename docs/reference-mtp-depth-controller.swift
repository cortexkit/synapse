import Foundation

/// Cost-model adaptive draft depth: pick the depth that maximizes expected
/// COMMITTED TOKENS PER WALL-CLOCK CYCLE, not the depth with the prettiest
/// acceptance streak.
///
/// `score(d) = (1 + p1 + p1*p2 + ... ) / t_est(d)` where `p_j` is the
/// probability draft row `j` is accepted (token-domain EMA) and `t_est(d)`
/// is the measured wall cost of a round at depth `d` (wall-clock-horizon
/// EMA). The numerator is the expected number of tokens a round commits
/// (the primary plus the expected accepted-prefix length); the denominator
/// is what a round actually costs on this machine right now, including
/// host bookkeeping, thermal state, and whatever else shares the GPU.
///
/// Design decisions ported from measured systems (omlx `_DepthController`,
/// via the MTPLX study), preserved because each one was bought with data:
///
/// - Acceptance is a TOKEN-domain EMA: it is a property of the model and
///   the content being decoded, so it updates per observed row.
/// - Cost is a WALL-CLOCK-horizon EMA (time constant `tauMs`): it tracks
///   context growth and machine state at constant real-time
///   responsiveness, and a one-off spike (a paging stall, a background
///   burst) is damped rather than swallowed whole.
/// - The marginal cost of one more verify row is the measured slope
///   between the cheapest and priciest depths seen so far, never a
///   constant guess.
/// - Probes are bidirectional and staleness-directed, duty-bounded: the
///   controller occasionally re-measures a rival depth, and re-measuring a
///   SHALLOWER depth matters most (a stale-high cost estimate for depth 1
///   would otherwise hide it forever).
/// - Switching has hysteresis so estimate noise does not thrash the depth.
///
/// The controller is deliberately ignorant of the scored window: it sees
/// only (attempted, accepted, elapsed) per round, which the session
/// already exposes.
final class Qwen36MTPDepthController {
    // Tuning constants. Values are the measured omlx defaults except
    // `marginalMsFallback`, scaled for a 27B target where a serial round
    // costs ~140-150ms on M1-class hardware (the fallback only matters
    // until the warmup ladder has measured two real depths).
    private let alpha = 0.08
    private let tauMs = 400.0
    private let probePeriodMs = 1000.0
    private let probePeriodMaxMs = 5000.0
    private let probeLen = 4
    private let probeDuty = 0.15
    private let probeMargin = 1.15
    private let spikeRatio = 2.0
    private let spikeDamp = 0.25
    private let marginalMsFallback = 20.0
    private let hysteresis = 1.03
    private let maxCycleMs = 5000.0

    private let maxDepth: Int
    private let minDepth: Int

    private(set) var currentDepth: Int
    private var p: [Double]
    private var t: [Int: Double] = [:]
    private var tAge: [Int: Double] = [:]
    private var probeLeft = 0
    private var msProbe = 0.0
    private var msExplore = 0.0
    private var warmup: [Int]
    private var lastObserve: TimeInterval?

    init(maxDepth: Int, minDepth: Int = 1) {
        precondition(maxDepth >= 1 && minDepth >= 1 && minDepth <= maxDepth)
        self.maxDepth = maxDepth
        self.minDepth = minDepth
        self.currentDepth = maxDepth
        // Optimistic acceptance prior; real observations overwrite it at
        // EMA speed. Starting optimistic makes the warmup ladder actually
        // exercise deep drafts instead of self-locking shallow.
        self.p = Array(repeating: 0.6, count: maxDepth)
        // One forced visit per depth, deepest first, so every depth owns a
        // real cost sample before the scorer is trusted.
        self.warmup = Array((minDepth...maxDepth).reversed())
    }

    // MARK: cost bookkeeping

    private func timeAlpha(_ cycleMs: Double) -> Double {
        1.0 - exp(-max(0.0, cycleMs) / tauMs)
    }

    private func updateTime(used: Int, cycleMs: Double) {
        guard let prev = t[used] else {
            t[used] = cycleMs
            return
        }
        if !warmup.isEmpty {
            t[used] = min(prev, cycleMs)
            return
        }
        var a = timeAlpha(cycleMs)
        if cycleMs > spikeRatio * prev { a *= spikeDamp }
        t[used] = (1.0 - a) * prev + a * cycleMs
    }

    private func marginalEstimate() -> Double {
        if t.count >= 2 {
            let depths = t.keys.sorted()
            let lo = depths.first!, hi = depths.last!
            if hi > lo {
                let slope = (t[hi]! - t[lo]!) / Double(hi - lo)
                if slope > 0 { return slope }
            }
        }
        return marginalMsFallback
    }

    private func tEstimate(_ d: Int) -> Double {
        if let known = t[d] { return known }
        guard !t.isEmpty else { return 140.0 + marginalMsFallback * Double(d) }
        let ref = t.keys.min(by: { abs($0 - d) < abs($1 - d) })!
        return max(1e-3, t[ref]! + marginalEstimate() * Double(d - ref))
    }

    private func score(_ d: Int) -> Double {
        var expected = 1.0
        var run = 1.0
        for j in 0..<d {
            run *= p[j]
            expected += run
        }
        return expected / max(1e-6, tEstimate(d))
    }

    // MARK: selection

    private var depthRange: ClosedRange<Int> { minDepth...maxDepth }

    private func best() -> Int {
        let curScore = score(currentDepth)
        var bestD = currentDepth
        var bestScore = curScore
        for d in depthRange where score(d) > bestScore {
            bestD = d
            bestScore = score(d)
        }
        if bestD != currentDepth && bestScore < curScore * hysteresis {
            return currentDepth
        }
        return bestD
    }

    private func bestRival() -> Int? {
        let bestScore = score(currentDepth)
        guard bestScore > 0 else { return mostStale() }
        var rival: Int?
        var rivalScore = 0.0
        for d in depthRange where d != currentDepth {
            let s = score(d)
            if s > rivalScore {
                rival = d
                rivalScore = s
            }
        }
        if let candidate = rival, rivalScore * probeMargin >= bestScore {
            return candidate
        }
        return nil
    }

    private func mostStale() -> Int? {
        let candidates = depthRange.filter { $0 != currentDepth }
        guard !candidates.isEmpty else { return nil }
        if let never = candidates.first(where: { t[$0] == nil }) { return never }
        return candidates.max(by: { tAge[$0, default: 0] < tAge[$1, default: 0] })
    }

    // MARK: the round interface

    /// Feed one completed round and receive the depth for the next one.
    func observe(attemptedDepth: Int, acceptedDepths: Int) -> Int {
        let now = ProcessInfo.processInfo.systemUptime
        var cycleMs: Double?
        if let last = lastObserve {
            let elapsed = (now - last) * 1000.0
            // An absurd inter-round gap is a stall or an interrupted run,
            // not a cycle cost; measuring it would poison the cost model.
            if elapsed > 0, elapsed <= maxCycleMs { cycleMs = elapsed }
        }
        lastObserve = now

        let used = max(1, min(attemptedDepth, maxDepth))
        let accepted = max(0, min(acceptedDepths, used))

        for j in 0..<used {
            let hit = j < accepted ? 1.0 : 0.0
            p[j] = (1.0 - alpha) * p[j] + alpha * hit
            if j >= accepted { break }
        }

        if let ms = cycleMs {
            updateTime(used: used, cycleMs: ms)
            for key in tAge.keys { tAge[key]! += ms }
            tAge[used] = 0.0
            msProbe += ms
            msExplore += ms
        }

        if !warmup.isEmpty {
            // A warmup slot is consumed only once its depth has a real
            // cost sample; the very first round has no prior timestamp to
            // diff against, so that round repeats its depth.
            if cycleMs != nil && used == warmup[0] { warmup.removeFirst() }
            if !warmup.isEmpty {
                currentDepth = warmup[0]
            } else {
                currentDepth = best()
                msProbe = 0.0
            }
        } else if probeLeft > 0 {
            probeLeft -= 1
            if probeLeft == 0 {
                currentDepth = best()
                msProbe = 0.0
            }
        } else {
            currentDepth = best()
            if maxDepth > minDepth, let ms = cycleMs {
                let period = max(probePeriodMs, Double(probeLen) * ms / probeDuty)
                if msProbe >= period {
                    let exploreDue = msExplore >= max(probePeriodMaxMs, 2.0 * period)
                    let target = exploreDue ? mostStale() : bestRival()
                    if let probeTarget = target {
                        currentDepth = probeTarget
                        probeLeft = probeLen
                        msProbe = 0.0
                        if exploreDue { msExplore = 0.0 }
                    }
                }
            }
        }
        return currentDepth
    }
}
