import AVFoundation
import Combine
import SwiftUI
import Vision

/// Orchestrator: camera → holistic landmarks → stream / offline → conversation UI + speech.
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
    @Published var latestBodyJoints: [LandmarkFrame.SerializedJoint] = []
    @Published var latestFaceJoints: [LandmarkFrame.SerializedJoint] = []
    @Published var lastErrorMessage: String?
    @Published var history: [ConversationTurn] = []
    @Published var liveSigningText: String = ""
    @Published var partialGloss: [String] = []
    @Published var speechEnabled: Bool = true

    let camera = CameraManager()
    let recognition = RecognitionClient()
    let speech = SpeechTranscriptController()
    let recorder = LandmarkRecorder()

    private let visionQueue = DispatchQueue(label: "com.aslsubtitles.vision", qos: .userInitiated)
    private let detector = HolisticPoseDetector()
    private let frameBuffer = LandmarkFrameBuffer(capacity: 36)
    private let segmenter = UtteranceSegmenter()

    private var frameCounter = 0
    private let processEveryNFrames = 2
    private var utteranceFrames: [LandmarkFrame] = []
    private var cancellables = Set<AnyCancellable>()

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

        recognition.onFinalSentence = { [weak self] text, confidence in
            guard let self else { return }
            self.appendSigningFinal(text: text, gloss: self.partialGloss, confidence: confidence)
        }
        recognition.onPartialUpdate = { [weak self] english, gloss, confidence in
            guard let self else { return }
            self.liveSigningText = english
            self.partialGloss = gloss
            self.smoothedSubtitle = english
            self.subtitleConfidence = confidence
            self.isWatchingEmpty = english.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        speech.onFinalUtterance = { [weak self] text in
            self?.appendSpeakingFinal(text: text)
        }

        if permissionStatus == .authorized {
            start()
        }
    }

    func requestPermissionAndStart() {
        camera.requestAccess { [weak self] granted in
            guard let self else { return }
            Task { @MainActor in
                self.permissionStatus = AVCaptureDevice.authorizationStatus(for: .video)
                if granted { self.start() }
            }
        }
    }

    func start() {
        visionQueue.async { [detector, frameBuffer, segmenter] in
            detector.reset()
            frameBuffer.reset()
            segmenter.reset()
        }
        recognition.resetOffline()
        utteranceFrames.removeAll()
        camera.configureAndStart(preferFront: false)
        isProcessing = true
        recognition.connect()
        if speechEnabled {
            Task {
                let ok = await speech.requestPermissions()
                if ok { speech.start() }
            }
        }
    }

    func stop() {
        camera.stop()
        speech.stop(cancel: true)
        recognition.disconnect()
        isProcessing = false
    }

    func toggleCamera() {
        camera.toggleCamera()
        visionQueue.async { [detector] in detector.reset() }
        recognition.resetOffline()
    }

    func toggleSpeech() {
        if speech.isListening {
            speech.stop(cancel: true)
            speechEnabled = false
        } else {
            speechEnabled = true
            Task {
                let ok = await speech.requestPermissions()
                if ok { speech.start() }
            }
        }
    }

    func setSpeechEnabled(_ enabled: Bool) {
        speechEnabled = enabled
        if enabled {
            Task {
                let ok = await speech.requestPermissions()
                if ok { speech.start() }
            }
        } else {
            speech.stop(cancel: true)
        }
    }

    func clearHistory() {
        history.removeAll()
        liveSigningText = ""
        partialGloss = []
    }

    private func appendSigningFinal(text: String, gloss: [String], confidence: Double) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        if let last = history.last, last.role == .signing, last.text == trimmed { return }
        history.append(ConversationTurn(role: .signing, text: trimmed, gloss: gloss, confidence: confidence))
        liveSigningText = trimmed
    }

    private func appendSpeakingFinal(text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        history.append(ConversationTurn(role: .speaking, text: trimmed))
    }

    private func handsFromFrame(_ frame: LandmarkFrame) -> [HandPoseSnapshot] {
        frame.hands.compactMap { serialized in
            var joints: [VNHumanHandPoseObservation.JointName: CGPoint] = [:]
            for (name, xy) in serialized.joints where xy.count >= 2 {
                if let j = Self.mapJoint(name) {
                    joints[j] = CGPoint(x: xy[0], y: xy[1])
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

    nonisolated fileprivate func processFrame(_ sampleBuffer: CMSampleBuffer, isFront: Bool) {
        visionQueue.async { [weak self, detector, frameBuffer, segmenter] in
            guard let self else { return }
            let frame = detector.detect(in: sampleBuffer, isFrontCamera: isFront)
            frameBuffer.append(frame)
            let event = segmenter.push(activity: frame.activity, at: frame.timestamp)
            let windowCopy = frameBuffer.window()
            let inUtterance = segmenter.isInUtterance

            Task { @MainActor in
                let hands = self.handsFromFrame(frame)
                self.latestHands = hands
                self.latestBodyJoints = frame.body
                self.latestFaceJoints = frame.face
                self.recognition.sendFrame(frame)
                self.recorder.append(frame)

                switch event {
                case .utteranceBegan:
                    self.utteranceFrames = [frame]
                case .utteranceEnded:
                    self.utteranceFrames.append(frame)
                    let frames = self.utteranceFrames.isEmpty ? windowCopy : self.utteranceFrames
                    self.recognition.endUtterance(frames: frames)
                    self.utteranceFrames.removeAll(keepingCapacity: true)
                case .none:
                    if inUtterance {
                        self.utteranceFrames.append(frame)
                        if self.utteranceFrames.count > 96 {
                            self.utteranceFrames.removeFirst(self.utteranceFrames.count - 96)
                        }
                    }
                }
            }
        }
    }

    fileprivate func shouldProcessFrame() -> Bool {
        frameCounter += 1
        return frameCounter % processEveryNFrames == 0
    }

    private static func mapJoint(_ name: String) -> VNHumanHandPoseObservation.JointName? {
        switch name {
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
