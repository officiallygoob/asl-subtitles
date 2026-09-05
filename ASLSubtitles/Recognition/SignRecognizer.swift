import Foundation
import Vision

/// Orchestrates on-device Core ML (primary) → everyday signs → fingerspelling,
/// then applies non-manual markers (NMMs) so face/body cue English subtitles.
///
/// Privacy: all inference stays on-device. No network required.
final class SignRecognizer {
    private let fingerspelling = FingerspellingClassifier()
    private let everyday = EverydaySignHeuristics()
    private let coreML = CoreMLSignClassifier()
    private let nmmAnalyzer = NonManualMarkersAnalyzer()

    private var previousCenters: [CGPoint] = []
    private var lastCenter: CGPoint?
    private var lateralDirection: CGFloat = 0
    private var lateralOscillations = 0
    private var framesSinceOscillationReset = 0
    /// Holistic FEATURE_DIM windows for Core ML (hands+face+body+NMM).
    private var featureWindow: [[Double]] = []
    private let windowSize = 32

    private(set) var lastNMM: NMMState = .zero

    var coreMLAvailable: Bool { coreML.isAvailable }
    var coreMLModelName: String { coreML.modelName }

    func reset() {
        previousCenters.removeAll()
        lastCenter = nil
        lateralDirection = 0
        lateralOscillations = 0
        framesSinceOscillationReset = 0
        featureWindow.removeAll()
        nmmAnalyzer.reset()
        lastNMM = .zero
    }

    /// Preferred entry: holistic frame so NMMs + Core ML see face/body grammar.
    func recognize(frame: LandmarkFrame) -> RecognitionResult {
        lastNMM = nmmAnalyzer.push(frame)
        var stamped = frame
        if stamped.nmm == nil {
            stamped.nmm = lastNMM.channelValues
        }
        featureWindow.append(stamped.featureVector())
        if featureWindow.count > windowSize {
            featureWindow.removeFirst(featureWindow.count - windowSize)
        }

        let hands = Self.hands(from: stamped)
        let raw: RecognitionResult
        if featureWindow.count >= 12, let ml = coreML.classify(window: featureWindow) {
            raw = ml
            if ml.confidence >= 0.55 {
                coreML.previousGloss = (ml.gloss.isEmpty ? ml.label : ml.gloss).uppercased()
            }
        } else {
            raw = recognizeHands(hands)
        }
        guard !raw.label.isEmpty else { return raw }
        let english = GlossEnglish.english(gloss: raw.gloss.isEmpty ? raw.label : raw.gloss, nmm: lastNMM)
        return RecognitionResult(
            label: english,
            kind: raw.kind,
            confidence: raw.confidence,
            timestamp: raw.timestamp,
            gloss: (raw.gloss.isEmpty ? raw.label : raw.gloss).uppercased(),
            nmm: lastNMM
        )
    }

    func recognize(hands: [HandPoseSnapshot]) -> RecognitionResult {
        recognizeHands(hands)
    }

    private func recognizeHands(_ hands: [HandPoseSnapshot]) -> RecognitionResult {
        guard !hands.isEmpty else {
            decayMotion()
            return .empty
        }

        let features = hands.map { LandmarkFeatures(hand: $0) }
        let motion = updateMotion(with: features[0])

        if let sign = everyday.classify(hands: features, motion: motion), sign.confidence >= 0.58 {
            return sign
        }

        if let letter = fingerspelling.classify(features[0]) {
            if let sign = everyday.classify(hands: features, motion: motion),
               sign.confidence >= letter.confidence {
                return sign
            }
            return letter
        }

        if let sign = everyday.classify(hands: features, motion: motion) {
            return sign
        }

        return .empty
    }

    private func updateMotion(with primary: LandmarkFeatures) -> EverydaySignHeuristics.MotionContext {
        let center = CGPoint(x: primary.handCenterX, y: primary.handCenterY)
        defer { lastCenter = center }

        guard let last = lastCenter else {
            return .init(deltaY: 0, deltaX: 0, lateralOscillations: 0)
        }

        let dx = center.x - last.x
        let dy = center.y - last.y

        framesSinceOscillationReset += 1
        if framesSinceOscillationReset > 45 {
            lateralOscillations = 0
            framesSinceOscillationReset = 0
        }

        if abs(dx) > 0.02 {
            let dir: CGFloat = dx > 0 ? 1 : -1
            if lateralDirection != 0 && dir != lateralDirection {
                lateralOscillations += 1
                framesSinceOscillationReset = 0
            }
            lateralDirection = dir
        }

        return .init(deltaY: dy, deltaX: dx, lateralOscillations: lateralOscillations)
    }

    private func decayMotion() {
        framesSinceOscillationReset += 1
        if framesSinceOscillationReset > 30 {
            lateralOscillations = 0
            lastCenter = nil
            lateralDirection = 0
        }
    }

    private static func hands(from frame: LandmarkFrame) -> [HandPoseSnapshot] {
        frame.hands.compactMap { serialized in
            var joints: [VNHumanHandPoseObservation.JointName: CGPoint] = [:]
            for (name, xy) in serialized.joints where xy.count >= 2 {
                if let j = mapJoint(name) {
                    joints[j] = CGPoint(x: xy[0], y: xy[1])
                }
            }
            guard !joints.isEmpty else { return nil }
            let chirality: VNChirality
            switch serialized.chirality {
            case "left": chirality = .left
            case "right": chirality = .right
            default: chirality = .unknown
            }
            return HandPoseSnapshot(chirality: chirality, joints: joints, confidence: serialized.confidence)
        }
    }

    private static func mapJoint(_ name: String) -> VNHumanHandPoseObservation.JointName? {
        switch name {
        case "wrist": return .wrist
        case "thumbCMC": return .thumbCMC
        case "thumbMP": return .thumbMP
        case "thumbIP": return .thumbIP
        case "thumbTip": return .thumbTip
        case "indexMCP": return .indexMCP
        case "indexPIP": return .indexPIP
        case "indexDIP": return .indexDIP
        case "indexTip": return .indexTip
        case "middleMCP": return .middleMCP
        case "middlePIP": return .middlePIP
        case "middleDIP": return .middleDIP
        case "middleTip": return .middleTip
        case "ringMCP": return .ringMCP
        case "ringPIP": return .ringPIP
        case "ringDIP": return .ringDIP
        case "ringTip": return .ringTip
        case "littleMCP": return .littleMCP
        case "littlePIP": return .littlePIP
        case "littleDIP": return .littleDIP
        case "littleTip": return .littleTip
        default: return nil
        }
    }
}
