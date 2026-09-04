import AVFoundation
import Combine
import SwiftUI

/// App-facing orchestrator: camera → Vision → recognition → smoothed subtitles.
@MainActor
final class ASLSessionController: NSObject, ObservableObject {
    @Published var permissionStatus: AVAuthorizationStatus = .notDetermined
    @Published var smoothedSubtitle: String = ""
    @Published var subtitleConfidence: Double = 0
    @Published var isWatchingEmpty: Bool = true
    @Published var isProcessing: Bool = false
    @Published var isFrontCamera: Bool = false
    @Published var showLandmarkDebug: Bool = false
    @Published var latestHands: [HandPoseSnapshot] = []
    @Published var lastErrorMessage: String?

    let camera = CameraManager()

    private let visionQueue = DispatchQueue(label: "com.aslsubtitles.vision", qos: .userInitiated)
    private let detector = HandPoseDetector()
    private let recognizer = SignRecognizer()
    private let smoother = TemporalSmoother()

    private var frameCounter = 0
    /// Process every Nth frame to keep Vision load reasonable.
    private let processEveryNFrames = 2

    var captureSession: AVCaptureSession { camera.session }

    func prepare() {
        permissionStatus = AVCaptureDevice.authorizationStatus(for: .video)
        camera.$permissionStatus
            .receive(on: DispatchQueue.main)
            .assign(to: &$permissionStatus)
        camera.$isFrontCamera
            .receive(on: DispatchQueue.main)
            .assign(to: &$isFrontCamera)
        camera.onConfigurationError = { [weak self] error in
            Task { @MainActor in
                self?.lastErrorMessage = error.localizedDescription
            }
        }
        camera.setFrameDelegate(self)

        if permissionStatus == .authorized {
            start()
        }
    }

    func requestPermissionAndStart() {
        camera.requestAccess { [weak self] granted in
            guard let self else { return }
            Task { @MainActor in
                self.permissionStatus = AVCaptureDevice.authorizationStatus(for: .video)
                if granted {
                    self.start()
                }
            }
        }
    }

    func start() {
        recognizer.reset()
        smoother.reset()
        camera.configureAndStart(preferFront: false)
        isProcessing = true
    }

    func stop() {
        camera.stop()
        isProcessing = false
    }

    func toggleCamera() {
        camera.toggleCamera()
        visionQueue.async { [recognizer] in
            recognizer.reset()
        }
    }

    /// Called from the camera frame queue; hops to visionQueue then MainActor for UI.
    nonisolated fileprivate func processFrame(_ sampleBuffer: CMSampleBuffer, isFront: Bool) {
        visionQueue.async { [detector, recognizer] in
            let hands = detector.detect(in: sampleBuffer, isFrontCamera: isFront)
            let result = recognizer.recognize(hands: hands)
            Task { @MainActor in
                self.latestHands = hands
                let smoothed = self.smoother.push(result)
                self.smoothedSubtitle = smoothed.text
                self.subtitleConfidence = smoothed.confidence
                self.isWatchingEmpty = smoothed.isEmpty
            }
        }
    }

    fileprivate func shouldProcessFrame() -> Bool {
        frameCounter += 1
        return frameCounter % processEveryNFrames == 0
    }
}

extension ASLSessionController: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        Task { @MainActor in
            guard self.shouldProcessFrame() else { return }
            let front = self.isFrontCamera
            self.processFrame(sampleBuffer, isFront: front)
        }
    }
}
