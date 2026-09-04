import AVFoundation
import QuartzCore
import Vision

/// On-device holistic landmark extractor: hands + body pose + face landmarks.
/// Inspired by MediaPipe Holistic / DeepMind SL2T landmark-only streaming —
/// we do **not** claim Google's model; we stream geometry only.
final class HolisticPoseDetector {
    private let handRequest: VNDetectHumanHandPoseRequest = {
        let r = VNDetectHumanHandPoseRequest()
        r.maximumHandCount = 2
        return r
    }()

    private let bodyRequest: VNDetectHumanBodyPoseRequest = {
        VNDetectHumanBodyPoseRequest()
    }()

    private let faceRequest: VNDetectFaceLandmarksRequest = {
        let r = VNDetectFaceLandmarksRequest()
        return r
    }()

    private let confidenceThreshold: Float = 0.25
    private var previousCenters: [CGPoint] = []
    private var sessionStart: CFTimeInterval?

    /// Legacy hand-only API used by offline heuristic fallback.
    func detectHands(in sampleBuffer: CMSampleBuffer, isFrontCamera: Bool) -> [HandPoseSnapshot] {
        let frame = detect(in: sampleBuffer, isFrontCamera: isFrontCamera)
        return frame.hands.compactMap { serialized -> HandPoseSnapshot? in
            var joints: [VNHumanHandPoseObservation.JointName: CGPoint] = [:]
            for (name, xy) in serialized.joints where xy.count >= 2 {
                if let joint = Self.jointName(from: name) {
                    joints[joint] = CGPoint(x: xy[0], y: xy[1])
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

    func detect(in sampleBuffer: CMSampleBuffer, isFrontCamera: Bool) -> LandmarkFrame {
        let now = CACurrentMediaTime()
        if sessionStart == nil { sessionStart = now }
        let timestamp = now - (sessionStart ?? now)

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return LandmarkFrame(timestamp: timestamp, hands: [], body: [], face: [], activity: 0)
        }

        let orientation: CGImagePropertyOrientation = isFrontCamera ? .leftMirrored : .right
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation, options: [:])

        do {
            try handler.perform([handRequest, bodyRequest, faceRequest])
        } catch {
            return LandmarkFrame(timestamp: timestamp, hands: [], body: [], face: [], activity: 0)
        }

        let hands = serializeHands(handRequest.results)
        let body = serializeBody(bodyRequest.results?.first)
        let face = serializeFace(faceRequest.results?.first)
        let activity = computeActivity(hands: hands, body: body)

        return LandmarkFrame(
            timestamp: timestamp,
            hands: hands,
            body: body,
            face: face,
            activity: activity
        )
    }

    func reset() {
        sessionStart = nil
        previousCenters = []
    }

    // MARK: - Serialization

    private func serializeHands(_ observations: [VNHumanHandPoseObservation]?) -> [LandmarkFrame.SerializedHand] {
        guard let observations else { return [] }
        return observations.compactMap { obs in
            guard obs.confidence >= confidenceThreshold else { return nil }
            var joints: [String: [Double]] = [:]
            for name in VNHumanHandPoseObservation.JointName.allTracked {
                guard let point = try? obs.recognizedPoint(name),
                      point.confidence >= confidenceThreshold else { continue }
                joints[Self.stringName(for: name)] = [Double(point.location.x), Double(point.location.y)]
            }
            guard !joints.isEmpty else { return nil }
            let chirality: String
            switch obs.chirality {
            case .left: chirality = "left"
            case .right: chirality = "right"
            default: chirality = "unknown"
            }
            return LandmarkFrame.SerializedHand(
                chirality: chirality,
                confidence: obs.confidence,
                joints: joints
            )
        }
    }

    private func serializeBody(_ observation: VNHumanBodyPoseObservation?) -> [LandmarkFrame.SerializedJoint] {
        guard let observation, observation.confidence >= confidenceThreshold else { return [] }
        var joints: [LandmarkFrame.SerializedJoint] = []
        let mapping: [(VNHumanBodyPoseObservation.JointName, String)] = [
            (.nose, "nose"),
            (.neck, "neck"),
            (.rightShoulder, "rightShoulder"),
            (.rightElbow, "rightElbow"),
            (.rightWrist, "rightWrist"),
            (.leftShoulder, "leftShoulder"),
            (.leftElbow, "leftElbow"),
            (.leftWrist, "leftWrist"),
            (.rightHip, "rightHip"),
            (.rightKnee, "rightKnee"),
            (.rightAnkle, "rightAnkle"),
            (.leftHip, "leftHip"),
            (.leftKnee, "leftKnee"),
            (.leftAnkle, "leftAnkle"),
            (.root, "root"),
            (.rightEar, "rightEar"),
            (.leftEar, "leftEar")
        ]
        for (joint, name) in mapping {
            guard let point = try? observation.recognizedPoint(joint),
                  point.confidence >= confidenceThreshold else { continue }
            joints.append(.init(
                name: name,
                x: Double(point.location.x),
                y: Double(point.location.y),
                confidence: point.confidence
            ))
        }
        return joints
    }

    private func serializeFace(_ observation: VNFaceObservation?) -> [LandmarkFrame.SerializedJoint] {
        guard let observation, let landmarks = observation.landmarks else { return [] }
        var joints: [LandmarkFrame.SerializedJoint] = []
        let box = observation.boundingBox

        func add(_ name: String, region: VNFaceLandmarkRegion2D?) {
            guard let region, let first = region.normalizedPoints.first else { return }
            // Face landmarks are relative to the face bounding box.
            let x = Double(box.origin.x + first.x * box.size.width)
            let y = Double(box.origin.y + first.y * box.size.height)
            joints.append(.init(name: name, x: x, y: y, confidence: observation.confidence))
        }

        add("leftEye", region: landmarks.leftEye)
        add("rightEye", region: landmarks.rightEye)
        add("nose", region: landmarks.noseCrest ?? landmarks.nose)
        add("mouthLeft", region: landmarks.innerLips)
        if let outer = landmarks.outerLips, outer.pointCount > 0 {
            let pts = outer.normalizedPoints
            let left = pts.min(by: { $0.x < $1.x })
            let right = pts.max(by: { $0.x < $1.x })
            if let left {
                joints.append(.init(
                    name: "mouthLeft",
                    x: Double(box.origin.x + left.x * box.size.width),
                    y: Double(box.origin.y + left.y * box.size.height),
                    confidence: observation.confidence
                ))
            }
            if let right {
                joints.append(.init(
                    name: "mouthRight",
                    x: Double(box.origin.x + right.x * box.size.width),
                    y: Double(box.origin.y + right.y * box.size.height),
                    confidence: observation.confidence
                ))
            }
            let mid = pts.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
            let n = CGFloat(pts.count)
            joints.append(.init(
                name: "mouthCenter",
                x: Double(box.origin.x + (mid.x / n) * box.size.width),
                y: Double(box.origin.y + (mid.y / n) * box.size.height),
                confidence: observation.confidence
            ))
        }
        add("leftEyebrowOuter", region: landmarks.leftEyebrow)
        add("rightEyebrowOuter", region: landmarks.rightEyebrow)
        add("chin", region: landmarks.faceContour)
        // Approximate forehead from median face box top.
        joints.append(.init(
            name: "forehead",
            x: Double(box.midX),
            y: Double(box.maxY),
            confidence: observation.confidence
        ))
        return joints
    }

    private func computeActivity(hands: [LandmarkFrame.SerializedHand], body: [LandmarkFrame.SerializedJoint]) -> Double {
        var centers: [CGPoint] = []
        for hand in hands {
            if let wrist = hand.joints["wrist"], wrist.count >= 2 {
                centers.append(CGPoint(x: wrist[0], y: wrist[1]))
            } else if let mcp = hand.joints["middleMCP"], mcp.count >= 2 {
                centers.append(CGPoint(x: mcp[0], y: mcp[1]))
            }
        }
        for name in ["leftWrist", "rightWrist"] {
            if let j = body.first(where: { $0.name == name }) {
                centers.append(CGPoint(x: j.x, y: j.y))
            }
        }

        defer { previousCenters = centers }

        guard !centers.isEmpty, !previousCenters.isEmpty else {
            return centers.isEmpty ? 0 : 0.2
        }

        var total: Double = 0
        let n = min(centers.count, previousCenters.count)
        for i in 0..<n {
            total += hypot(Double(centers[i].x - previousCenters[i].x),
                           Double(centers[i].y - previousCenters[i].y))
        }
        let mean = total / Double(max(n, 1))
        // ~0.02 normalized units/frame ≈ moderate motion
        return min(1.0, mean / 0.04)
    }

    // MARK: - Name maps

    private static func stringName(for joint: VNHumanHandPoseObservation.JointName) -> String {
        switch joint {
        case .wrist: return "wrist"
        case .thumbCMC: return "thumbCMC"
        case .thumbMP: return "thumbMP"
        case .thumbIP: return "thumbIP"
        case .thumbTip: return "thumbTip"
        case .indexMCP: return "indexMCP"
        case .indexPIP: return "indexPIP"
        case .indexDIP: return "indexDIP"
        case .indexTip: return "indexTip"
        case .middleMCP: return "middleMCP"
        case .middlePIP: return "middlePIP"
        case .middleDIP: return "middleDIP"
        case .middleTip: return "middleTip"
        case .ringMCP: return "ringMCP"
        case .ringPIP: return "ringPIP"
        case .ringDIP: return "ringDIP"
        case .ringTip: return "ringTip"
        case .littleMCP: return "littleMCP"
        case .littlePIP: return "littlePIP"
        case .littleDIP: return "littleDIP"
        case .littleTip: return "littleTip"
        default: return "unknown"
        }
    }

    private static func jointName(from string: String) -> VNHumanHandPoseObservation.JointName? {
        switch string {
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
