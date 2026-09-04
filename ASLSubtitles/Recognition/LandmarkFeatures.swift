import CoreGraphics
import Foundation
import Vision

/// Geometry helpers derived from a single-hand landmark set.
///
/// TODO(CoreML): Replace heuristic distances / angles with a fixed-length
/// feature vector exported to a Core ML classifier (e.g. Create ML Hand Pose).
struct LandmarkFeatures {
    let hand: HandPoseSnapshot

    // MARK: Finger extended?

    func isExtended(_ tip: VNHumanHandPoseObservation.JointName,
                    pip: VNHumanHandPoseObservation.JointName,
                    mcp: VNHumanHandPoseObservation.JointName) -> Bool {
        guard let t = hand.point(tip), let p = hand.point(pip), let m = hand.point(mcp) else {
            return false
        }
        // Tip farther from wrist than PIP, and roughly collinear outward.
        guard let wrist = hand.point(.wrist) else { return false }
        let tipDist = distance(t, wrist)
        let pipDist = distance(p, wrist)
        let mcpDist = distance(m, wrist)
        return tipDist > pipDist * 1.05 && pipDist > mcpDist * 0.95
    }

    var thumbExtended: Bool {
        guard let tip = hand.point(.thumbTip),
              let ip = hand.point(.thumbIP),
              let mcp = hand.point(.thumbMP),
              let wrist = hand.point(.wrist) else { return false }
        return distance(tip, wrist) > distance(ip, wrist) && distance(ip, wrist) > distance(mcp, wrist) * 0.9
    }

    var indexExtended: Bool { isExtended(.indexTip, pip: .indexPIP, mcp: .indexMCP) }
    var middleExtended: Bool { isExtended(.middleTip, pip: .middlePIP, mcp: .middleMCP) }
    var ringExtended: Bool { isExtended(.ringTip, pip: .ringPIP, mcp: .ringMCP) }
    var littleExtended: Bool { isExtended(.littleTip, pip: .littlePIP, mcp: .littleMCP) }

    var extendedFingerCount: Int {
        [thumbExtended, indexExtended, middleExtended, ringExtended, littleExtended]
            .filter { $0 }.count
    }

    var nonThumbExtendedCount: Int {
        [indexExtended, middleExtended, ringExtended, littleExtended].filter { $0 }.count
    }

    /// Approximate "fist" — no non-thumb fingers clearly extended.
    var isFist: Bool { nonThumbExtendedCount == 0 && !indexExtended }

    /// Open palm — most fingers extended.
    var isOpenPalm: Bool { nonThumbExtendedCount >= 3 && extendedFingerCount >= 4 }

    func distance(_ a: CGPoint, _ b: CGPoint) -> CGFloat {
        hypot(a.x - b.x, a.y - b.y)
    }

    func tip(_ name: VNHumanHandPoseObservation.JointName) -> CGPoint? {
        hand.point(name)
    }

    /// Angle at B formed by points A–B–C, in degrees.
    func angle(a: CGPoint, b: CGPoint, c: CGPoint) -> CGFloat {
        let ab = CGPoint(x: a.x - b.x, y: a.y - b.y)
        let cb = CGPoint(x: c.x - b.x, y: c.y - b.y)
        let dot = ab.x * cb.x + ab.y * cb.y
        let mag = hypot(ab.x, ab.y) * hypot(cb.x, cb.y)
        guard mag > 1e-6 else { return 0 }
        let cos = max(-1, min(1, dot / mag))
        return acos(cos) * 180 / .pi
    }

    /// Thumb tip near index tip (pinch / "O"-like).
    var thumbIndexPinch: Bool {
        guard let t = tip(.thumbTip), let i = tip(.indexTip) else { return false }
        return distance(t, i) < 0.06
    }

    /// Tips clustered near palm center (closed hand).
    var tipsNearPalm: Bool {
        guard let index = tip(.indexTip),
              let middle = tip(.middleTip),
              let mcp = hand.point(.middleMCP) else { return false }
        return distance(index, mcp) < 0.12 && distance(middle, mcp) < 0.12
    }

    var handCenterY: CGFloat {
        guard let wrist = hand.point(.wrist), let mcp = hand.point(.middleMCP) else {
            return hand.point(.wrist)?.y ?? 0.5
        }
        return (wrist.y + mcp.y) / 2
    }

    var handCenterX: CGFloat {
        guard let wrist = hand.point(.wrist), let mcp = hand.point(.middleMCP) else {
            return 0.5
        }
        return (wrist.x + mcp.x) / 2
    }

    /// Normalized feature vector placeholder for a future Core ML model.
    /// TODO(CoreML): Feed this (or an expanded set) into an MLModel.
    func featureVector() -> [Double] {
        var values: [Double] = []
        let order = VNHumanHandPoseObservation.JointName.allTracked
        guard let origin = hand.point(.wrist) else {
            return Array(repeating: 0, count: order.count * 2 + 5)
        }
        for joint in order {
            if let p = hand.point(joint) {
                values.append(Double(p.x - origin.x))
                values.append(Double(p.y - origin.y))
            } else {
                values.append(0)
                values.append(0)
            }
        }
        values.append(thumbExtended ? 1 : 0)
        values.append(indexExtended ? 1 : 0)
        values.append(middleExtended ? 1 : 0)
        values.append(ringExtended ? 1 : 0)
        values.append(littleExtended ? 1 : 0)
        return values
    }
}
