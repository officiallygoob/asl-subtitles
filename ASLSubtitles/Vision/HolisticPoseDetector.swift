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
        let conf = observation.confidence

        func toImage(_ p: CGPoint) -> (Double, Double) {
            (
                Double(box.origin.x + p.x * box.size.width),
                Double(box.origin.y + p.y * box.size.height)
            )
        }

        func addPoint(_ name: String, _ p: CGPoint) {
            let (x, y) = toImage(p)
            joints.append(.init(name: name, x: x, y: y, confidence: conf))
        }

        func addFirst(_ name: String, region: VNFaceLandmarkRegion2D?) {
            guard let region, let first = region.normalizedPoints.first else { return }
            addPoint(name, first)
        }

        func regionPts(_ region: VNFaceLandmarkRegion2D?) -> [CGPoint] {
            guard let region, region.pointCount > 0 else { return [] }
            return region.normalizedPoints
        }

        // Eyes — centroid + vertical extremes for openness.
        if let pts = Optional(regionPts(landmarks.leftEye)), !pts.isEmpty {
            let mid = pts.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
            let n = CGFloat(pts.count)
            addPoint("leftEye", CGPoint(x: mid.x / n, y: mid.y / n))
            if let top = pts.max(by: { $0.y < $1.y }) { addPoint("leftEyeTop", top) }
            if let bot = pts.min(by: { $0.y < $1.y }) { addPoint("leftEyeBottom", bot) }
        }
        if let pts = Optional(regionPts(landmarks.rightEye)), !pts.isEmpty {
            let mid = pts.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
            let n = CGFloat(pts.count)
            addPoint("rightEye", CGPoint(x: mid.x / n, y: mid.y / n))
            if let top = pts.max(by: { $0.y < $1.y }) { addPoint("rightEyeTop", top) }
            if let bot = pts.min(by: { $0.y < $1.y }) { addPoint("rightEyeBottom", bot) }
        }

        addFirst("nose", region: landmarks.noseCrest ?? landmarks.nose)

        // Eyebrows — outer + inner extremes.
        if let pts = Optional(regionPts(landmarks.leftEyebrow)), !pts.isEmpty {
            if let outer = pts.min(by: { $0.x < $1.x }) { addPoint("leftEyebrowOuter", outer) }
            if let inner = pts.max(by: { $0.x < $1.x }) { addPoint("leftEyebrowInner", inner) }
        }
        if let pts = Optional(regionPts(landmarks.rightEyebrow)), !pts.isEmpty {
            if let outer = pts.max(by: { $0.x < $1.x }) { addPoint("rightEyebrowOuter", outer) }
            if let inner = pts.min(by: { $0.x < $1.x }) { addPoint("rightEyebrowInner", inner) }
        }

        // Outer lips — left/right/center + top/bottom for mouth open / smile proxies.
        let outerPts = regionPts(landmarks.outerLips)
        if !outerPts.isEmpty {
            if let left = outerPts.min(by: { $0.x < $1.x }) { addPoint("mouthLeft", left) }
            if let right = outerPts.max(by: { $0.x < $1.x }) { addPoint("mouthRight", right) }
            if let top = outerPts.max(by: { $0.y < $1.y }) { addPoint("outerLipTop", top) }
            if let bot = outerPts.min(by: { $0.y < $1.y }) { addPoint("outerLipBottom", bot) }
            let mid = outerPts.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
            let n = CGFloat(outerPts.count)
            addPoint("mouthCenter", CGPoint(x: mid.x / n, y: mid.y / n))
        }

        let innerPts = regionPts(landmarks.innerLips)
        if !innerPts.isEmpty {
            if let top = innerPts.max(by: { $0.y < $1.y }) { addPoint("innerLipTop", top) }
            if let bot = innerPts.min(by: { $0.y < $1.y }) { addPoint("innerLipBottom", bot) }
            if joints.first(where: { $0.name == "mouthCenter" }) == nil {
                let mid = innerPts.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
                let n = CGFloat(innerPts.count)
                addPoint("mouthCenter", CGPoint(x: mid.x / n, y: mid.y / n))
            }
        }

        // Chin from face contour lowest point; forehead from box top.
        let contour = regionPts(landmarks.faceContour)
        if let chin = contour.min(by: { $0.y < $1.y }) {
            addPoint("chin", chin)
        } else {
            addFirst("chin", region: landmarks.faceContour)
        }
        joints.append(.init(
            name: "forehead",
            x: Double(box.midX),
            y: Double(box.maxY),
            confidence: conf
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
