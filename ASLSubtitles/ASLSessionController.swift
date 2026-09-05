import AVFoundation
import Combine
import SwiftUI
import UIKit
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
    @Published var latestNMM: NMMState = .zero
    @Published var showNMMBadges: Bool = true
    @Published var suggestedReplies: [String] = []
    @Published var conversationSummary: String = ""
    @Published var isSummarizing: Bool = false
    @Published var showHistorySheet: Bool = false
    /// Stable id for the in-progress conversation (persisted on-device).
    @Published private(set) var conversationID: UUID = UUID()
    @Published var lastErrorMessage: String?
    @Published var history: [ConversationTurn] = []
    /// Accumulating English transcript for the signing side (persists across empty gaps).
    @Published var persistentSigningTranscript: String = ""
    /// In-progress / live word from the recognizer (may clear when detection fades).
    @Published var currentSigningWord: String = ""
    /// Legacy single-slot caption; kept in sync with current word for older call sites.
    @Published var liveSigningText: String = ""
    @Published var partialGloss: [String] = []
    @Published var speechEnabled: Bool = true

    let camera = CameraManager()
    let recognition = RecognitionClient()
    let speech = SpeechTranscriptController()
    let recorder = LandmarkRecorder()
    let callMode = CallModeController()
    let sharePlay = SharePlayCoordinator()
    /// Optional frame sink for Learn tab (nil when not learning).
    var learnConsumer: ((LandmarkFrame) -> Void)?

    private let visionQueue = DispatchQueue(label: "com.aslsubtitles.vision", qos: .userInitiated)
    private let detector = HolisticPoseDetector()
    private let frameBuffer = LandmarkFrameBuffer(capacity: 36)
    private let segmenter = UtteranceSegmenter()
    private let nmmAnalyzer = NonManualMarkersAnalyzer()

    private var frameCounter = 0
    private let processEveryNFrames = 2
    private var utteranceFrames: [LandmarkFrame] = []
    private var cancellables = Set<AnyCancellable>()

    var captureSession: AVCaptureSession { camera.session }

    /// Caption area: persistent transcript + optional in-progress word when not already at the end.
    var captionDisplayText: String {
        let persistent = persistentSigningTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        let current = currentSigningWord.trimmingCharacters(in: .whitespacesAndNewlines)
        if persistent.isEmpty {
            return current
        }
        if current.isEmpty {
            return persistent
        }
        let lastToken = persistent.split(separator: " ").last.map(String.init) ?? ""
        if lastToken.compare(current, options: [.caseInsensitive, .diacriticInsensitive]) == .orderedSame {
            return persistent
        }
        return persistent + " " + current
    }

    func prepare() {
        ASLAppBridge.bind(self)
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
            let trimmed = english.trimmingCharacters(in: .whitespacesAndNewlines)
            // Never wipe the persistent transcript when live detection goes empty.
            self.currentSigningWord = trimmed
            self.liveSigningText = trimmed
            self.partialGloss = gloss
            self.smoothedSubtitle = trimmed
            self.subtitleConfidence = confidence
            self.isWatchingEmpty = self.persistentSigningTranscript.isEmpty && trimmed.isEmpty
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
        visionQueue.async { [detector, frameBuffer, segmenter, nmmAnalyzer] in
            detector.reset()
            frameBuffer.reset()
            segmenter.reset()
            nmmAnalyzer.reset()
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
        persistentSigningTranscript = ""
        currentSigningWord = ""
        partialGloss = []
        smoothedSubtitle = ""
        subtitleConfidence = 0
        isWatchingEmpty = true
        suggestedReplies = []
        conversationSummary = ""
        conversationID = UUID()
    }

    private func appendSigningFinal(text: String, gloss: [String], confidence: Double) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        Task { @MainActor in
            let polished = await AppleIntelligenceText.polishEnglish(trimmed)
            self.commitSigningFinal(text: polished, gloss: gloss, confidence: confidence)
            self.sharePlay.broadcastCaption(polished)
            self.refreshSuggestedReplies(after: polished)
        }
    }

    private func commitSigningFinal(text: String, gloss: [String], confidence: Double) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        let lastToken = persistentSigningTranscript
            .split(separator: " ")
            .last
            .map(String.init) ?? ""
        if lastToken.compare(trimmed, options: [.caseInsensitive, .diacriticInsensitive]) != .orderedSame {
            if persistentSigningTranscript.isEmpty {
                persistentSigningTranscript = trimmed
            } else {
                persistentSigningTranscript += " " + trimmed
            }
        }

        if let last = history.last, last.role == .signing, last.text == trimmed {
            currentSigningWord = trimmed
            liveSigningText = trimmed
            persistConversation()
            return
        }
        history.append(ConversationTurn(role: .signing, text: trimmed, gloss: gloss, confidence: confidence))
        currentSigningWord = trimmed
        liveSigningText = trimmed
        persistConversation()
    }

    func refreshSuggestedReplies(after english: String) {
        Task { @MainActor in
            let hist = self.history.suffix(8).map { (role: $0.role.rawValue, text: $0.text) }
            let suggestions = await SuggestedReplies.suggest(afterSigning: english, history: Array(hist))
            self.suggestedReplies = suggestions
        }
    }

    func applySuggestedReply(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        history.append(ConversationTurn(role: .speaking, text: trimmed))
        suggestedReplies = []
        persistConversation()
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }

    func requestConversationSummary() {
        guard !history.isEmpty || !persistentSigningTranscript.isEmpty else {
            conversationSummary = ""
            return
        }
        isSummarizing = true
        Task { @MainActor in
            let turns = self.history.map { (role: $0.role == .signing ? "Signing" : "You said", text: $0.text) }
            let summary = await AppleIntelligenceText.summarizeConversation(turns)
            self.conversationSummary = summary
            self.isSummarizing = false
        }
    }

    /// Sync placeholder for Siri while async summary runs.
    func summarizeConversationSyncPlaceholder() -> String {
        if !conversationSummary.isEmpty { return conversationSummary }
        if !persistentSigningTranscript.isEmpty { return persistentSigningTranscript }
        return history.suffix(4).map(\.text).joined(separator: " · ")
    }

    func persistConversation() {
        ConversationStore.shared.upsertLive(
            id: conversationID,
            turns: history,
            signingTranscript: persistentSigningTranscript
        )
    }

    func openPersistedConversation(id: UUID) {
        guard let record = ConversationStore.shared.conversations.first(where: { $0.id == id }) else { return }
        conversationID = record.id
        persistentSigningTranscript = record.signingTranscript
        history = record.turns.map { t in
            ConversationTurn(
                role: t.role == "Signing" ? .signing : (t.role == "You said" ? .speaking : .system),
                text: t.text,
                gloss: t.gloss,
                confidence: t.confidence,
                timestamp: t.createdAt
            )
        }
        currentSigningWord = ""
        liveSigningText = ""
        conversationSummary = ""
        suggestedReplies = []
        showHistorySheet = false
    }

    func startNewConversation() {
        persistConversation()
        conversationID = UUID()
        clearHistory()
    }

    private func appendSpeakingFinal(text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        history.append(ConversationTurn(role: .speaking, text: trimmed))
        persistConversation()
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

    func startCallModeFromIntent() {
        prepare()
        if permissionStatus == .authorized { start() }
        beginCallMode()
    }

    func beginCallMode() {
        callMode.onSampleBuffer = { [weak self] buffer in
            guard let self else { return }
            // Treat captured FaceTime frames like camera frames (front=false for external content).
            self.processFrame(buffer, isFront: false)
        }
        callMode.startCallMode()
        sharePlay.observeSessions()
        // Poll broadcast extension captions / heartbeat
        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] timer in
            Task { @MainActor in
                guard let self else { timer.invalidate(); return }
                guard self.callMode.isCallModeActive else { timer.invalidate(); return }
                if let caption = self.callMode.pollBroadcastCaptions(), !caption.isEmpty {
                    // Heartbeat only — real captions come from Vision on SCK frames.
                    _ = caption
                }
            }
        }
    }

    func endCallMode() {
        callMode.stopCallMode()
    }

    nonisolated fileprivate func processFrame(_ sampleBuffer: CMSampleBuffer, isFront: Bool) {
        visionQueue.async { [weak self, detector, frameBuffer, segmenter, nmmAnalyzer] in
            guard let self else { return }
            var frame = detector.detect(in: sampleBuffer, isFrontCamera: isFront)
            let nmm = nmmAnalyzer.push(frame, window: frameBuffer.window(n: 12))
            frame.nmm = nmm.channelValues
            frame.featureLayoutVersion = LandmarkFrame.featureLayoutVersion
            frameBuffer.append(frame)
            let event = segmenter.push(activity: frame.activity, at: frame.timestamp)
            let windowCopy = frameBuffer.window()
            let inUtterance = segmenter.isInUtterance

            Task { @MainActor in
                let hands = self.handsFromFrame(frame)
                self.latestHands = hands
                self.latestBodyJoints = frame.body
                self.latestFaceJoints = frame.face
                self.latestNMM = nmm
                self.recognition.sendFrame(frame, nmm: nmm)
                self.recorder.append(frame)
                self.learnConsumer?(frame)

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
