import AVFoundation
import Vision

/// Runs VNDetectHumanHandPoseRequest on camera frames (on-device only).
final class HandPoseDetector {
    private let request: VNDetectHumanHandPoseRequest = {
        let r = VNDetectHumanHandPoseRequest()
        r.maximumHandCount = 2
        return r
    }()

    private let confidenceThreshold: Float = 0.3

    func detect(in sampleBuffer: CMSampleBuffer, isFrontCamera: Bool) -> [HandPoseSnapshot] {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return [] }

        let orientation: CGImagePropertyOrientation = isFrontCamera ? .leftMirrored : .right
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation, options: [:])

        do {
            try handler.perform([request])
        } catch {
            return []
        }

        guard let observations = request.results else { return [] }

        return observations.compactMap { observation in
            guard observation.confidence >= confidenceThreshold else { return nil }

            var joints: [VNHumanHandPoseObservation.JointName: CGPoint] = [:]
            for name in VNHumanHandPoseObservation.JointName.allTracked {
                guard let point = try? observation.recognizedPoint(name),
                      point.confidence >= confidenceThreshold else { continue }
                joints[name] = CGPoint(x: point.location.x, y: point.location.y)
            }

            guard !joints.isEmpty else { return nil }

            return HandPoseSnapshot(
                chirality: observation.chirality,
                joints: joints,
                confidence: observation.confidence
            )
        }
    }
}
