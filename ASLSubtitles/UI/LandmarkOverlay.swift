import SwiftUI
import Vision

/// Optional debug overlay drawing hand joints and bones.
struct LandmarkOverlay: View {
    let hands: [HandPoseSnapshot]

    private let bones: [(VNHumanHandPoseObservation.JointName, VNHumanHandPoseObservation.JointName)] = [
        (.wrist, .thumbCMC), (.thumbCMC, .thumbMP), (.thumbMP, .thumbIP), (.thumbIP, .thumbTip),
        (.wrist, .indexMCP), (.indexMCP, .indexPIP), (.indexPIP, .indexDIP), (.indexDIP, .indexTip),
        (.wrist, .middleMCP), (.middleMCP, .middlePIP), (.middlePIP, .middleDIP), (.middleDIP, .middleTip),
        (.wrist, .ringMCP), (.ringMCP, .ringPIP), (.ringPIP, .ringDIP), (.ringDIP, .ringTip),
        (.wrist, .littleMCP), (.littleMCP, .littlePIP), (.littlePIP, .littleDIP), (.littleDIP, .littleTip),
        (.indexMCP, .middleMCP), (.middleMCP, .ringMCP), (.ringMCP, .littleMCP)
    ]

    var body: some View {
        GeometryReader { geo in
            Canvas { context, size in
                for hand in hands {
                    for (a, b) in bones {
                        guard let pa = hand.overlayPoint(a, in: size),
                              let pb = hand.overlayPoint(b, in: size) else { continue }
                        var path = Path()
                        path.move(to: pa)
                        path.addLine(to: pb)
                        context.stroke(path, with: .color(.orange.opacity(0.85)), lineWidth: 2.5)
                    }
                    for joint in VNHumanHandPoseObservation.JointName.allTracked {
                        guard let p = hand.overlayPoint(joint, in: size) else { continue }
                        let rect = CGRect(x: p.x - 4, y: p.y - 4, width: 8, height: 8)
                        context.fill(Path(ellipseIn: rect), with: .color(.yellow))
                    }
                }
            }
            .frame(width: geo.size.width, height: geo.size.height)
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}
