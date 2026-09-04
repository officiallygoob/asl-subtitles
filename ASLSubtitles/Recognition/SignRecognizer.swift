import Foundation
import Vision

/// Orchestrates Core ML (when present) → everyday signs → fingerspelling.
///
/// Offline fallback path only — Conversation Mode prefers the recognition server.
final class SignRecognizer {
    private let fingerspelling = FingerspellingClassifier()
    private let everyday = EverydaySignHeuristics()
    private let coreML = CoreMLSignClassifier()

    private var previousCenters: [CGPoint] = []
    private var lastCenter: CGPoint?
    private var lateralDirection: CGFloat = 0
    private var lateralOscillations = 0
    private var framesSinceOscillationReset = 0
    private var featureWindow: [[Double]] = []
    private let windowSize = 24

    var coreMLAvailable: Bool { coreML.isAvailable }
    var coreMLModelName: String { coreML.modelName }

    func reset() {
        previousCenters.removeAll()
        lastCenter = nil
        lateralDirection = 0
        lateralOscillations = 0
        framesSinceOscillationReset = 0
        featureWindow.removeAll()
    }

    func recognize(hands: [HandPoseSnapshot]) -> RecognitionResult {
        guard !hands.isEmpty else {
            decayMotion()
            featureWindow.removeAll()
            return .empty
        }

        let features = hands.map { LandmarkFeatures(hand: $0) }
        let motion = updateMotion(with: features[0])

        // Maintain a short temporal window for Core ML Hand Action-style models.
        featureWindow.append(features[0].featureVector())
        if featureWindow.count > windowSize {
            featureWindow.removeFirst(featureWindow.count - windowSize)
        }

        // Prefer Core ML when a user-supplied model is confident.
        if featureWindow.count >= 12, let ml = coreML.classify(window: featureWindow) {
            return ml
        }

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
}
