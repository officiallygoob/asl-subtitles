import CoreGraphics
import Foundation

/// Estimated non-manual markers (NMMs) from face + body landmarks.
/// Phone Vision landmarks are approximate vs studio MoCap — treat as soft cues.
struct NMMState: Equatable, Codable {
    var browRaise: Double = 0
    var browFurrow: Double = 0
    var eyeWiden: Double = 0
    var squint: Double = 0
    var mouthOpen: Double = 0
    var smile: Double = 0
    var frown: Double = 0
    var headShake: Double = 0
    var headNod: Double = 0
    var torsoLean: Double = 0
    var shoulderTilt: Double = 0
    /// Overall confidence that face geometry was present enough to estimate.
    var confidence: Double = 0

    static let zero = NMMState()

    /// Ordered channel names matching `LandmarkFrame.nmmChannelOrder` / server NMM_CHANNELS.
    var channelValues: [Double] {
        [
            browRaise, browFurrow, eyeWiden, squint,
            mouthOpen, smile, frown,
            headShake, headNod,
            torsoLean, shoulderTilt
        ]
    }

    var isQuestionLikely: Bool { browRaise >= 0.38 && confidence >= 0.28 }
    var isNegationLikely: Bool { (headShake >= 0.38 || frown >= 0.48) && confidence >= 0.25 }
    var isEmphasisLikely: Bool { abs(torsoLean) >= 0.28 || shoulderTilt >= 0.32 }

    /// Compact badges for debug overlay.
    var activeBadges: [String] {
        var badges: [String] = []
        if browRaise >= 0.4 { badges.append("brow↑") }
        if browFurrow >= 0.4 { badges.append("brow↓") }
        if eyeWiden >= 0.45 { badges.append("eyes") }
        if squint >= 0.45 { badges.append("squint") }
        if mouthOpen >= 0.45 { badges.append("mouth") }
        if smile >= 0.45 { badges.append("smile") }
        if frown >= 0.45 { badges.append("frown") }
        if headShake >= 0.4 { badges.append("shake") }
        if headNod >= 0.4 { badges.append("nod") }
        if abs(torsoLean) >= 0.35 { badges.append(torsoLean > 0 ? "lean→" : "lean←") }
        if shoulderTilt >= 0.4 { badges.append("tilt") }
        return badges
    }

    func summaryDict() -> [String: Double] {
        [
            "browRaise": browRaise,
            "browFurrow": browFurrow,
            "eyeWiden": eyeWiden,
            "squint": squint,
            "mouthOpen": mouthOpen,
            "smile": smile,
            "frown": frown,
            "headShake": headShake,
            "headNod": headNod,
            "torsoLean": torsoLean,
            "shoulderTilt": shoulderTilt,
            "confidence": confidence
        ]
    }
}

/// Rolling analyzer: single-frame geometry + short temporal buffer for head shake/nod.
final class NonManualMarkersAnalyzer {
    private var noseHistory: [(t: TimeInterval, x: Double, y: Double)] = []
    private let historySeconds: TimeInterval = 0.55
    private var lastState = NMMState.zero

    func reset() {
        noseHistory.removeAll(keepingCapacity: true)
        lastState = .zero
    }

    /// Update from the latest frame (optionally a short window for stability).
    @discardableResult
    func push(_ frame: LandmarkFrame, window: [LandmarkFrame] = []) -> NMMState {
        let faceMap = Dictionary(uniqueKeysWithValues: frame.face.map { ($0.name, $0) })
        let bodyMap = Dictionary(uniqueKeysWithValues: frame.body.map { ($0.name, $0) })

        var state = NMMState()
        let facePresent = faceMap.count >= 4
        state.confidence = facePresent ? min(1.0, Double(faceMap.count) / 12.0) : 0.15

        // --- Brows ---
        // Raise: eyebrow y above eye (Vision y increases upward).
        if let lb = faceMap["leftEyebrowOuter"] ?? faceMap["leftEyebrowInner"],
           let le = faceMap["leftEye"] {
            let raise = max(0, lb.y - le.y - 0.012)
            let furrow = max(0, le.y - lb.y - 0.004)
            state.browRaise = clamp01(raise / 0.045)
            state.browFurrow = clamp01(furrow / 0.03)
        }
        if let rb = faceMap["rightEyebrowOuter"] ?? faceMap["rightEyebrowInner"],
           let re = faceMap["rightEye"] {
            let raise = max(0, rb.y - re.y - 0.012)
            let furrow = max(0, re.y - rb.y - 0.004)
            state.browRaise = max(state.browRaise, clamp01(raise / 0.045))
            state.browFurrow = max(state.browFurrow, clamp01(furrow / 0.03))
        }

        // --- Eyes ---
        let leftOpen = eyeAperture(top: faceMap["leftEyeTop"], bottom: faceMap["leftEyeBottom"], fallback: faceMap["leftEye"])
        let rightOpen = eyeAperture(top: faceMap["rightEyeTop"], bottom: faceMap["rightEyeBottom"], fallback: faceMap["rightEye"])
        let eyeOpen = max(leftOpen, rightOpen)
        state.eyeWiden = clamp01((eyeOpen - 0.028) / 0.04)
        state.squint = clamp01((0.018 - eyeOpen) / 0.015)

        // --- Mouth ---
        let mouthH = verticalSpan(
            top: faceMap["outerLipTop"] ?? faceMap["innerLipTop"],
            bottom: faceMap["outerLipBottom"] ?? faceMap["innerLipBottom"],
            fallbackA: faceMap["mouthCenter"],
            fallbackB: faceMap["chin"]
        )
        state.mouthOpen = clamp01((mouthH - 0.02) / 0.06)

        if let ml = faceMap["mouthLeft"], let mr = faceMap["mouthRight"], let mc = faceMap["mouthCenter"] {
            let width = abs(mr.x - ml.x)
            let lift = mc.y - min(ml.y, mr.y)
            // Smile: wider mouth + corners lifted relative to center.
            state.smile = clamp01((width - 0.06) / 0.08) * clamp01((lift + 0.01) / 0.03)
            let drop = min(ml.y, mr.y) - mc.y
            state.frown = clamp01((drop - 0.005) / 0.025) * (1.0 - state.smile)
        }

        // --- Head shake / nod from nose/ear temporal buffer ---
        if let nose = faceMap["nose"] ?? bodyMap["nose"] {
            noseHistory.append((frame.timestamp, nose.x, nose.y))
            let cutoff = frame.timestamp - historySeconds
            noseHistory.removeAll { $0.t < cutoff }
            let (shake, nod) = headMotionProxies(noseHistory)
            state.headShake = shake
            state.headNod = nod
        } else if let earL = bodyMap["leftEar"], let earR = bodyMap["rightEar"] {
            let midX = (earL.x + earR.x) * 0.5
            let midY = (earL.y + earR.y) * 0.5
            noseHistory.append((frame.timestamp, midX, midY))
            let cutoff = frame.timestamp - historySeconds
            noseHistory.removeAll { $0.t < cutoff }
            let (shake, nod) = headMotionProxies(noseHistory)
            state.headShake = shake
            state.headNod = nod
        }

        // --- Torso lean / shoulder tilt ---
        let lean = LandmarkFrame.torsoLean(from: frame.body)
        let tilt = LandmarkFrame.shoulderTilt(from: frame.body)
        state.torsoLean = lean
        state.shoulderTilt = abs(tilt)

        // Mild temporal EMA so badges don't flicker.
        lastState = blend(lastState, state, alpha: 0.45)
        return lastState
    }

    private func eyeAperture(
        top: LandmarkFrame.SerializedJoint?,
        bottom: LandmarkFrame.SerializedJoint?,
        fallback: LandmarkFrame.SerializedJoint?
    ) -> Double {
        if let top, let bottom {
            return abs(top.y - bottom.y)
        }
        // Without eyelid points, use a neutral mid aperture so we don't invent widen/squint.
        _ = fallback
        return 0.022
    }

    private func verticalSpan(
        top: LandmarkFrame.SerializedJoint?,
        bottom: LandmarkFrame.SerializedJoint?,
        fallbackA: LandmarkFrame.SerializedJoint?,
        fallbackB: LandmarkFrame.SerializedJoint?
    ) -> Double {
        if let top, let bottom {
            return abs(top.y - bottom.y)
        }
        if let a = fallbackA, let b = fallbackB {
            return abs(a.y - b.y) * 0.45
        }
        return 0.015
    }

    private func headMotionProxies(_ hist: [(t: TimeInterval, x: Double, y: Double)]) -> (Double, Double) {
        guard hist.count >= 5 else { return (0, 0) }
        let xs = hist.map(\.x)
        let ys = hist.map(\.y)
        var xSignChanges = 0
        var ySignChanges = 0
        for i in 2..<xs.count {
            let dx0 = xs[i - 1] - xs[i - 2]
            let dx1 = xs[i] - xs[i - 1]
            if dx0 * dx1 < 0 && abs(dx1) > 0.004 { xSignChanges += 1 }
            let dy0 = ys[i - 1] - ys[i - 2]
            let dy1 = ys[i] - ys[i - 1]
            if dy0 * dy1 < 0 && abs(dy1) > 0.004 { ySignChanges += 1 }
        }
        let xRange = (xs.max() ?? 0) - (xs.min() ?? 0)
        let yRange = (ys.max() ?? 0) - (ys.min() ?? 0)
        let shake = clamp01(Double(xSignChanges) / 4.0) * clamp01(xRange / 0.04)
        let nod = clamp01(Double(ySignChanges) / 4.0) * clamp01(yRange / 0.035)
        return (shake, nod)
    }

    private func blend(_ a: NMMState, _ b: NMMState, alpha: Double) -> NMMState {
        func mix(_ x: Double, _ y: Double) -> Double { x * (1 - alpha) + y * alpha }
        return NMMState(
            browRaise: mix(a.browRaise, b.browRaise),
            browFurrow: mix(a.browFurrow, b.browFurrow),
            eyeWiden: mix(a.eyeWiden, b.eyeWiden),
            squint: mix(a.squint, b.squint),
            mouthOpen: mix(a.mouthOpen, b.mouthOpen),
            smile: mix(a.smile, b.smile),
            frown: mix(a.frown, b.frown),
            headShake: mix(a.headShake, b.headShake),
            headNod: mix(a.headNod, b.headNod),
            torsoLean: mix(a.torsoLean, b.torsoLean),
            shoulderTilt: mix(a.shoulderTilt, b.shoulderTilt),
            confidence: mix(a.confidence, b.confidence)
        )
    }

    private func clamp01(_ v: Double) -> Double { min(1, max(0, v)) }
}
