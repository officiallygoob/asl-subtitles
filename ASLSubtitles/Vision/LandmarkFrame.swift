import CoreGraphics
import Foundation
import Vision

/// One temporal frame of holistic landmarks (hands + body + face).
/// Video pixels are discarded — only geometry is kept / streamed.
///
/// ## Feature vector layout (protocol v2 / featureLayoutVersion 2)
/// | Slice            | Dim | Notes |
/// |------------------|-----|-------|
/// | left hand 21×2   | 42  | unchanged |
/// | right hand 21×2  | 42  | unchanged |
/// | body 17×2        | 34  | unchanged |
/// | face 20×2        | 40  | v1 had 10×2; **additive** joints appended |
/// | activity         | 1   | unchanged |
/// | NMM channels     | 11  | **additive** brow/eye/mouth/head/torso proxies |
/// | **FEATURE_DIM**  | **170** | was 139 in v1 |
struct LandmarkFrame: Codable, Equatable, Identifiable {
    /// Bump when featureVector dimensionality / layout changes.
    static let featureLayoutVersion = 2
    static let featureDim = 21 * 2 * 2 + 17 * 2 + 20 * 2 + 1 + 11

    var id: UUID = UUID()
    /// Monotonic capture time (seconds since session start preferred; wall clock OK).
    var timestamp: TimeInterval
    var hands: [SerializedHand]
    var body: [SerializedJoint]
    var face: [SerializedJoint]
    /// Rough activity score 0…1 used for utterance segmentation.
    var activity: Double
    /// Optional precomputed NMM channel values (0…1); when nil, packed as zeros
    /// and the analyzer / server derive from face+body.
    var nmm: [Double]? = nil
    /// Wire protocol / feature layout version stamped on outbound frames.
    var featureLayoutVersion: Int = LandmarkFrame.featureLayoutVersion

    struct SerializedHand: Codable, Equatable {
        /// "left" | "right" | "unknown"
        var chirality: String
        var confidence: Float
        /// Joint name → normalized (x, y) in Vision coords (origin bottom-left).
        var joints: [String: [Double]]
    }

    struct SerializedJoint: Codable, Equatable {
        var name: String
        var x: Double
        var y: Double
        var confidence: Float
    }

    /// Flat feature vector suitable for sequence models (fixed length 170).
    func featureVector() -> [Double] {
        var values: [Double] = []
        values += Self.packHand(named: "left", in: hands)
        values += Self.packHand(named: "right", in: hands)
        values += Self.packJoints(body, expected: Self.bodyJointOrder)
        values += Self.packJoints(face, expected: Self.faceJointOrder)
        values.append(activity)
        let nmmVals = nmm ?? Array(repeating: 0, count: Self.nmmChannelOrder.count)
        if nmmVals.count >= Self.nmmChannelOrder.count {
            values += Array(nmmVals.prefix(Self.nmmChannelOrder.count))
        } else {
            values += nmmVals + Array(repeating: 0, count: Self.nmmChannelOrder.count - nmmVals.count)
        }
        return values
    }

    static let handJointOrder = [
        "wrist",
        "thumbCMC", "thumbMP", "thumbIP", "thumbTip",
        "indexMCP", "indexPIP", "indexDIP", "indexTip",
        "middleMCP", "middlePIP", "middleDIP", "middleTip",
        "ringMCP", "ringPIP", "ringDIP", "ringTip",
        "littleMCP", "littlePIP", "littleDIP", "littleTip"
    ]

    static let bodyJointOrder = [
        "nose", "neck",
        "rightShoulder", "rightElbow", "rightWrist",
        "leftShoulder", "leftElbow", "leftWrist",
        "rightHip", "rightKnee", "rightAnkle",
        "leftHip", "leftKnee", "leftAnkle",
        "root", "rightEar", "leftEar"
    ]

    /// First 10 match protocol v1; remaining are additive (v2).
    static let faceJointOrder = [
        // v1
        "leftEye", "rightEye", "nose", "mouthLeft", "mouthRight",
        "leftEyebrowOuter", "rightEyebrowOuter",
        "chin", "forehead", "mouthCenter",
        // v2 additive — brows / lids / lips for NMM proxies
        "leftEyebrowInner", "rightEyebrowInner",
        "leftEyeTop", "leftEyeBottom", "rightEyeTop", "rightEyeBottom",
        "outerLipTop", "outerLipBottom", "innerLipTop", "innerLipBottom"
    ]

    static let nmmChannelOrder = [
        "browRaise", "browFurrow", "eyeWiden", "squint",
        "mouthOpen", "smile", "frown",
        "headShake", "headNod",
        "torsoLean", "shoulderTilt"
    ]

    // MARK: - Body helpers

    /// Signed torso lean in roughly −1…1 (positive = lean toward viewer-right / +x).
    static func torsoLean(from body: [SerializedJoint]) -> Double {
        let map = Dictionary(uniqueKeysWithValues: body.map { ($0.name, $0) })
        guard let ls = map["leftShoulder"], let rs = map["rightShoulder"],
              let lh = map["leftHip"], let rh = map["rightHip"] else { return 0 }
        let shoulderMidX = (ls.x + rs.x) * 0.5
        let hipMidX = (lh.x + rh.x) * 0.5
        let shoulderWidth = max(abs(rs.x - ls.x), 0.05)
        return max(-1, min(1, (shoulderMidX - hipMidX) / shoulderWidth))
    }

    /// Absolute shoulder-line tilt 0…1 (0 = level).
    static func shoulderTilt(from body: [SerializedJoint]) -> Double {
        let map = Dictionary(uniqueKeysWithValues: body.map { ($0.name, $0) })
        guard let ls = map["leftShoulder"], let rs = map["rightShoulder"] else { return 0 }
        let dy = abs(ls.y - rs.y)
        let dx = max(abs(rs.x - ls.x), 0.05)
        let angle = abs(atan2(dy, dx)) // radians
        return min(1, angle / 0.45)
    }

    private static func packHand(named chirality: String, in hands: [SerializedHand]) -> [Double] {
        if let hand = hands.first(where: { $0.chirality == chirality }) {
            return handJointOrder.flatMap { name -> [Double] in
                if let xy = hand.joints[name], xy.count >= 2 {
                    return [xy[0], xy[1]]
                }
                return [0, 0]
            }
        }
        return Array(repeating: 0, count: handJointOrder.count * 2)
    }

    private static func packJoints(_ joints: [SerializedJoint], expected: [String]) -> [Double] {
        let map = Dictionary(uniqueKeysWithValues: joints.map { ($0.name, $0) })
        return expected.flatMap { name -> [Double] in
            if let j = map[name] { return [j.x, j.y] }
            return [0, 0]
        }
    }
}

/// Rolling temporal buffer (~1–2 s at ~24–30 fps).
final class LandmarkFrameBuffer {
    private var frames: [LandmarkFrame] = []
    let capacity: Int

    init(capacity: Int = 36) {
        self.capacity = max(8, capacity)
    }

    var count: Int { frames.count }
    var allFrames: [LandmarkFrame] { frames }

    func append(_ frame: LandmarkFrame) {
        frames.append(frame)
        if frames.count > capacity {
            frames.removeFirst(frames.count - capacity)
        }
    }

    func reset() {
        frames.removeAll(keepingCapacity: true)
    }

    /// Most recent window (up to `n` frames).
    func window(n: Int? = nil) -> [LandmarkFrame] {
        let take = n ?? capacity
        guard frames.count > take else { return frames }
        return Array(frames.suffix(take))
    }

    /// Mean activity over the last `seconds`.
    func recentActivity(seconds: TimeInterval = 0.35) -> Double {
        guard let last = frames.last else { return 0 }
        let cutoff = last.timestamp - seconds
        let slice = frames.filter { $0.timestamp >= cutoff }
        guard !slice.isEmpty else { return 0 }
        return slice.map(\.activity).reduce(0, +) / Double(slice.count)
    }
}

/// Detects pause / rest-pose to end a signed utterance.
final class UtteranceSegmenter {
    private let restThreshold: Double
    private let restDuration: TimeInterval
    private let minUtterance: TimeInterval

    private var restStartedAt: TimeInterval?
    private var utteranceStartedAt: TimeInterval?
    private(set) var isInUtterance = false

    init(restThreshold: Double = 0.055, restDuration: TimeInterval = 0.75, minUtterance: TimeInterval = 0.50) {
        self.restThreshold = restThreshold
        self.restDuration = restDuration
        self.minUtterance = minUtterance
    }

    enum Event {
        case none
        case utteranceBegan
        case utteranceEnded
    }

    func reset() {
        restStartedAt = nil
        utteranceStartedAt = nil
        isInUtterance = false
    }

    /// Feed per-frame activity; returns segmentation events.
    func push(activity: Double, at timestamp: TimeInterval) -> Event {
        if activity >= restThreshold {
            restStartedAt = nil
            if !isInUtterance {
                isInUtterance = true
                utteranceStartedAt = timestamp
                return .utteranceBegan
            }
            return .none
        }

        // Low activity — potential rest.
        if restStartedAt == nil {
            restStartedAt = timestamp
        }
        guard isInUtterance,
              let restStart = restStartedAt,
              let utterStart = utteranceStartedAt else {
            return .none
        }
        if timestamp - restStart >= restDuration,
           timestamp - utterStart >= minUtterance {
            isInUtterance = false
            utteranceStartedAt = nil
            restStartedAt = nil
            return .utteranceEnded
        }
        return .none
    }
}
