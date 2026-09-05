import SwiftUI
import Vision

/// Debug overlay for hands + body + face landmarks, plus optional NMM badges.
struct HolisticLandmarkOverlay: View {
    let hands: [HandPoseSnapshot]
    let bodyJoints: [LandmarkFrame.SerializedJoint]
    let face: [LandmarkFrame.SerializedJoint]
    var nmm: NMMState? = nil

    private let handBones: [(VNHumanHandPoseObservation.JointName, VNHumanHandPoseObservation.JointName)] = [
        (.wrist, .thumbCMC), (.thumbCMC, .thumbMP), (.thumbMP, .thumbIP), (.thumbIP, .thumbTip),
        (.wrist, .indexMCP), (.indexMCP, .indexPIP), (.indexPIP, .indexDIP), (.indexDIP, .indexTip),
        (.wrist, .middleMCP), (.middleMCP, .middlePIP), (.middlePIP, .middleDIP), (.middleDIP, .middleTip),
        (.wrist, .ringMCP), (.ringMCP, .ringPIP), (.ringPIP, .ringDIP), (.ringDIP, .ringTip),
        (.wrist, .littleMCP), (.littleMCP, .littlePIP), (.littlePIP, .littleDIP), (.littleDIP, .littleTip),
        (.indexMCP, .middleMCP), (.middleMCP, .ringMCP), (.ringMCP, .littleMCP)
    ]

    private let bodyBones: [(String, String)] = [
        ("leftShoulder", "rightShoulder"),
        ("leftShoulder", "leftElbow"), ("leftElbow", "leftWrist"),
        ("rightShoulder", "rightElbow"), ("rightElbow", "rightWrist"),
        ("leftShoulder", "leftHip"), ("rightShoulder", "rightHip"),
        ("leftHip", "rightHip"),
        ("leftHip", "leftKnee"), ("leftKnee", "leftAnkle"),
        ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
        ("neck", "nose")
    ]

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .topLeading) {
                Canvas { context, size in
                    drawHands(context: context, size: size)
                    drawBodyJoints(context: context, size: size)
                    drawFace(context: context, size: size)
                }
                .frame(width: geo.size.width, height: geo.size.height)

                if let nmm, !nmm.activeBadges.isEmpty {
                    nmmBadgeRow(nmm)
                        .padding(12)
                }
            }
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }

    private func nmmBadgeRow(_ nmm: NMMState) -> some View {
        HStack(spacing: 6) {
            ForEach(nmm.activeBadges, id: \.self) { badge in
                Text(badge)
                    .font(.caption2.weight(.bold).monospaced())
                    .foregroundStyle(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(AppTheme.accent.opacity(0.55), in: Capsule())
            }
        }
    }

    private func drawHands(context: GraphicsContext, size: CGSize) {
        for hand in hands {
            for (a, b) in handBones {
                guard let pa = hand.overlayPoint(a, in: size),
                      let pb = hand.overlayPoint(b, in: size) else { continue }
                var path = Path()
                path.move(to: pa)
                path.addLine(to: pb)
                context.stroke(path, with: .color(.orange.opacity(0.85)), lineWidth: 2.5)
            }
            for joint in VNHumanHandPoseObservation.JointName.allTracked {
                guard let p = hand.overlayPoint(joint, in: size) else { continue }
                let rect = CGRect(x: p.x - 3.5, y: p.y - 3.5, width: 7, height: 7)
                context.fill(Path(ellipseIn: rect), with: .color(.yellow))
            }
        }
    }

    private func drawBodyJoints(context: GraphicsContext, size: CGSize) {
        let map = Dictionary(uniqueKeysWithValues: bodyJoints.map { ($0.name, $0) })
        func pt(_ name: String) -> CGPoint? {
            guard let j = map[name] else { return nil }
            return CGPoint(x: j.x * size.width, y: (1 - j.y) * size.height)
        }
        for (a, b) in bodyBones {
            guard let pa = pt(a), let pb = pt(b) else { continue }
            var path = Path()
            path.move(to: pa)
            path.addLine(to: pb)
            context.stroke(path, with: .color(.mint.opacity(0.8)), lineWidth: 2)
        }
        for j in bodyJoints {
            let p = CGPoint(x: j.x * size.width, y: (1 - j.y) * size.height)
            let rect = CGRect(x: p.x - 3, y: p.y - 3, width: 6, height: 6)
            context.fill(Path(ellipseIn: rect), with: .color(.mint))
        }
    }

    private func drawFace(context: GraphicsContext, size: CGSize) {
        for j in face {
            let p = CGPoint(x: j.x * size.width, y: (1 - j.y) * size.height)
            let rect = CGRect(x: p.x - 2.5, y: p.y - 2.5, width: 5, height: 5)
            context.fill(Path(ellipseIn: rect), with: .color(.pink.opacity(0.9)))
        }
    }
}
