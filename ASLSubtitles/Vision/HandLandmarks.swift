import CoreGraphics
import Vision

/// Normalized hand pose snapshot used across recognition and debug UI.
struct HandPoseSnapshot: Identifiable, Equatable {
    let id = UUID()
    let chirality: VNChirality
    /// Joint name → normalized image point (Vision coords: origin bottom-left).
    let joints: [VNHumanHandPoseObservation.JointName: CGPoint]
    let confidence: Float

    static func == (lhs: HandPoseSnapshot, rhs: HandPoseSnapshot) -> Bool {
        lhs.id == rhs.id
    }

    func point(_ joint: VNHumanHandPoseObservation.JointName) -> CGPoint? {
        joints[joint]
    }

    /// Convert Vision (bottom-left) to SwiftUI overlay (top-left) space.
    func overlayPoint(_ joint: VNHumanHandPoseObservation.JointName, in size: CGSize) -> CGPoint? {
        guard let p = joints[joint] else { return nil }
        return CGPoint(x: p.x * size.width, y: (1 - p.y) * size.height)
    }
}

extension VNHumanHandPoseObservation.JointName {
    static let allTracked: [VNHumanHandPoseObservation.JointName] = [
        .wrist,
        .thumbCMC, .thumbMP, .thumbIP, .thumbTip,
        .indexMCP, .indexPIP, .indexDIP, .indexTip,
        .middleMCP, .middlePIP, .middleDIP, .middleTip,
        .ringMCP, .ringPIP, .ringDIP, .ringTip,
        .littleMCP, .littlePIP, .littleDIP, .littleTip
    ]
}
