import AVFoundation
import Combine
import Foundation
import Speech

/// Live speech → text so YOU can speak and THEY can read (reverse conversation channel).
@MainActor
final class SpeechTranscriptController: NSObject, ObservableObject {
    @Published private(set) var authStatus: SFSpeechRecognizerAuthorizationStatus = .notDetermined
    @Published private(set) var isListening = false
    @Published private(set) var partialTranscript = ""
    @Published private(set) var lastFinalTranscript = ""
    @Published var micPermissionGranted = false
    @Published var lastError: String?

    var onFinalUtterance: ((String) -> Void)?

    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var audioEngine: AVAudioEngine?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?

    override init() {
        super.init()
        authStatus = SFSpeechRecognizer.authorizationStatus()
        micPermissionGranted = AVAudioSession.sharedInstance().recordPermission == .granted
    }

    func requestPermissions() async -> Bool {
        let speechOK: Bool = await withCheckedContinuation { cont in
            SFSpeechRecognizer.requestAuthorization { status in
                Task { @MainActor in
                    self.authStatus = status
                    cont.resume(returning: status == .authorized)
                }
            }
        }
        let micOK: Bool = await withCheckedContinuation { cont in
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                Task { @MainActor in
                    self.micPermissionGranted = granted
                    cont.resume(returning: granted)
                }
            }
        }
        return speechOK && micOK
    }

    func start() {
        guard !isListening else { return }
        guard authStatus == .authorized, micPermissionGranted else {
            lastError = "Microphone or speech recognition not authorized"
            return
        }
        guard let speechRecognizer, speechRecognizer.isAvailable else {
            lastError = "Speech recognizer unavailable"
            return
        }

        stop(cancel: true)

        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker, .allowBluetooth])
            try session.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            lastError = error.localizedDescription
            return
        }

        let engine = AVAudioEngine()
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.addsPunctuation = true
        if speechRecognizer.supportsOnDeviceRecognition {
            request.requiresOnDeviceRecognition = false // allow server ASR for better quality when available
        }

        recognitionRequest = request
        audioEngine = engine

        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
            request.append(buffer)
        }

        recognitionTask = speechRecognizer.recognitionTask(with: request) { [weak self] result, error in
            Task { @MainActor in
                guard let self else { return }
                if let result {
                    let text = result.bestTranscription.formattedString
                    self.partialTranscript = text
                    if result.isFinal {
                        self.lastFinalTranscript = text
                        self.partialTranscript = text
                        if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            self.onFinalUtterance?(text)
                        }
                    }
                }
                if let error {
                    // End-of-speech / cancel are normal.
                    let ns = error as NSError
                    if ns.domain == "kAFAssistantErrorDomain" && (ns.code == 1110 || ns.code == 1101) {
                        // no-op
                    } else if self.isListening {
                        self.lastError = error.localizedDescription
                    }
                    self.restartIfNeeded()
                } else if result?.isFinal == true {
                    self.restartIfNeeded()
                }
            }
        }

        do {
            try engine.start()
            isListening = true
            lastError = nil
        } catch {
            lastError = error.localizedDescription
            stop(cancel: true)
        }
    }

    func stop(cancel: Bool = false) {
        recognitionRequest?.endAudio()
        if cancel {
            recognitionTask?.cancel()
        } else {
            recognitionTask?.finish()
        }
        recognitionTask = nil
        recognitionRequest = nil
        if let engine = audioEngine {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }
        audioEngine = nil
        isListening = false
    }

    private func restartIfNeeded() {
        // Continuous conversation: restart recognition after each final segment.
        guard isListening || audioEngine != nil else { return }
        let keepGoing = isListening
        stop(cancel: false)
        partialTranscript = ""
        if keepGoing {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) { [weak self] in
                self?.start()
            }
        }
    }
}
