import Foundation
import Vision

/// Orchestrates landmark features → everyday signs → fingerspelling.
///
/// Extension point for Core ML:
/// 1. Train a model on `LandmarkFeatures.featureVector()` (+ optional motion).
/// 2. Add a `CoreMLSignClassifier` conforming to the same classify API.
/// 3. Prefer ML scores when confidence > heuristic confidence.
final class SignRecognizer {
    private let fingerspelling = FingerspellingClassifier()
    private let everyday = EverydaySignHeuristics()

    private var previousCenters: [CGPoint] = []
    private var lastCenter: CGPoint?
    private var lateralDirection: CGFloat = 0
    private var lateralOscillations = 0
    private var framesSinceOscillationReset = 0

    func reset() {
        previousCenters.removeAll()
        lastCenter = nil
        lateralDirection = 0
        lateralOscillations = 0
        framesSinceOscillationReset = 0
    }

    func recognize(hands: [HandPoseSnapshot]) -> RecognitionResult {
        guard !hands.isEmpty else {
            decayMotion()
            return .empty
        }

        let features = hands.map { LandmarkFeatures(hand: $0) }
        let motion = updateMotion(with: features[0])

        // Prefer multi-hand / motion everyday signs when confident.
        if let sign = everyday.classify(hands: features, motion: motion), sign.confidence >= 0.58 {
            return sign
        }

        // Fall back to fingerspelling on primary hand.
        if let letter = fingerspelling.classify(features[0]) {
            // Everyday signs win ties when both fire.
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

    // MARK: - Motion

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

        return .init(
            deltaY: dy,
            deltaX: dx,
            lateralOscillations: lateralOscillations
        )
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
