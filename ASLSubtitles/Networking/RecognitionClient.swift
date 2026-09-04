import Foundation
import Vision

/// Wire protocol messages (client → server / server → client).
/// Privacy: only landmark geometry is sent — never camera pixels.
enum RecognitionWire {
    struct ClientHello: Codable {
        var type: String = "hello"
        var protocolVersion: Int = 1
        var client: String = "ASLSubtitles-iOS"
        var sampleRateHz: Double
        var windowFrames: Int
    }

    struct FrameMessage: Codable {
        var type: String = "frame"
        var frame: LandmarkFrame
    }

    struct UtteranceEnd: Codable {
        var type: String = "utterance_end"
        var frames: [LandmarkFrame]
    }

    struct PartialResult: Codable {
        var type: String
        var gloss: [String]
        var english: String
        var confidence: Double
        var isFinal: Bool
        var source: String?
    }

    struct ServerStatus: Codable {
        var type: String
        var connected: Bool?
        var model: String?
        var message: String?
    }
}

enum RecognitionConnectionState: Equatable {
    case disconnected
    case connecting
    case connected
    case offlineFallback
    case error(String)
}

/// Streams landmark frames to a recognition server over WebSocket.
/// Falls back to on-device heuristics when the server is unavailable.
@MainActor
final class RecognitionClient: ObservableObject {
    @Published private(set) var state: RecognitionConnectionState = .disconnected
    @Published private(set) var partialEnglish: String = ""
    @Published private(set) var partialGloss: [String] = []
    @Published private(set) var lastFinalEnglish: String = ""
    @Published private(set) var lastConfidence: Double = 0
    @Published private(set) var modelName: String = ""

    var serverURLString: String {
        get { UserDefaults.standard.string(forKey: "recognitionServerURL") ?? Self.defaultURL }
        set {
            UserDefaults.standard.set(newValue, forKey: "recognitionServerURL")
            reconnect()
        }
    }

    var preferServer: Bool {
        get {
            if UserDefaults.standard.object(forKey: "preferRecognitionServer") == nil { return true }
            return UserDefaults.standard.bool(forKey: "preferRecognitionServer")
        }
        set {
            UserDefaults.standard.set(newValue, forKey: "preferRecognitionServer")
            if newValue { reconnect() } else { disconnect(); state = .offlineFallback }
        }
    }

    static let defaultURL = "ws://127.0.0.1:8765/v1/stream"

    private var task: URLSessionWebSocketTask?
    private var session: URLSession?
    private var pingTimer: Timer?
    private var reconnectAttempt = 0
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    /// Offline fallback recognizer (heuristics) — not the primary path.
    private let offlineRecognizer = SignRecognizer()
    private let offlineSmoother = TemporalSmoother()

    var onFinalSentence: ((String, Double) -> Void)?
    var onPartialUpdate: ((String, [String], Double) -> Void)?

    func connect() {
        guard preferServer else {
            state = .offlineFallback
            return
        }
        disconnect(clearState: false)
        guard let url = URL(string: serverURLString),
              let scheme = url.scheme,
              scheme == "ws" || scheme == "wss" else {
            state = .error("Invalid server URL")
            state = .offlineFallback
            return
        }

        state = .connecting
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 8
        let session = URLSession(configuration: config)
        self.session = session
        let task = session.webSocketTask(with: url)
        self.task = task
        task.resume()

        let hello = RecognitionWire.ClientHello(sampleRateHz: 15, windowFrames: 36)
        send(hello)

        receiveLoop()
        schedulePing()

        // Optimistic: mark connected; errors will flip to offline.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { [weak self] in
            guard let self, self.task != nil else { return }
            if case .connecting = self.state {
                self.state = .connected
                self.reconnectAttempt = 0
            }
        }
    }

    func reconnect() {
        connect()
    }

    func disconnect(clearState: Bool = true) {
        pingTimer?.invalidate()
        pingTimer = nil
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
        session?.invalidateAndCancel()
        session = nil
        if clearState {
            state = .disconnected
        }
    }

    func resetOffline() {
        offlineRecognizer.reset()
        offlineSmoother.reset()
    }

    /// Stream one landmark frame (primary path when connected).
    func sendFrame(_ frame: LandmarkFrame) {
        switch state {
        case .connected:
            send(RecognitionWire.FrameMessage(frame: frame))
        case .offlineFallback, .error, .disconnected, .connecting:
            applyOffline(frame)
        }
    }

    /// Notify server that a signed utterance ended; also flush offline.
    func endUtterance(frames: [LandmarkFrame]) {
        switch state {
        case .connected:
            send(RecognitionWire.UtteranceEnd(frames: frames))
        default:
            break
        }
    }

    // MARK: - Offline fallback

    private func applyOffline(_ frame: LandmarkFrame) {
        // Reconstruct HandPoseSnapshot for legacy heuristics.
        let hands: [HandPoseSnapshot] = frame.hands.compactMap { serialized in
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
        let result = offlineRecognizer.recognize(hands: hands)
        let smoothed = offlineSmoother.push(result)
        partialEnglish = smoothed.text
        partialGloss = smoothed.text.isEmpty ? [] : [smoothed.text.lowercased()]
        lastConfidence = smoothed.confidence
        onPartialUpdate?(partialEnglish, partialGloss, lastConfidence)
        if !smoothed.isEmpty, smoothed.confidence >= 0.7 {
            // Heuristics emit single glosses; treat stable labels as soft finals.
        }
    }

    // MARK: - Transport

    private func send<T: Encodable>(_ value: T) {
        guard let task else { return }
        do {
            let data = try encoder.encode(value)
            guard let text = String(data: data, encoding: .utf8) else { return }
            task.send(.string(text)) { [weak self] error in
                if let error {
                    Task { @MainActor in
                        self?.handleTransportError(error)
                    }
                }
            }
        } catch {
            // Encoding failures are local bugs; ignore.
        }
    }

    private func receiveLoop() {
        task?.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .failure(let error):
                Task { @MainActor in
                    self.handleTransportError(error)
                }
            case .success(let message):
                Task { @MainActor in
                    self.handleMessage(message)
                    self.receiveLoop()
                }
            }
        }
    }

    private func handleMessage(_ message: URLSessionWebSocketTask.Message) {
        let data: Data?
        switch message {
        case .string(let text): data = text.data(using: .utf8)
        case .data(let d): data = d
        @unknown default: data = nil
        }
        guard let data else { return }

        if let partial = try? decoder.decode(RecognitionWire.PartialResult.self, from: data),
           partial.type == "partial" || partial.type == "final" {
            partialGloss = partial.gloss
            partialEnglish = partial.english
            lastConfidence = partial.confidence
            onPartialUpdate?(partial.english, partial.gloss, partial.confidence)
            if partial.isFinal || partial.type == "final" {
                lastFinalEnglish = partial.english
                onFinalSentence?(partial.english, partial.confidence)
            }
            if state != .connected { state = .connected }
            return
        }

        if let status = try? decoder.decode(RecognitionWire.ServerStatus.self, from: data) {
            if let model = status.model { modelName = model }
            if status.type == "welcome" || status.type == "status" {
                state = .connected
            }
        }
    }

    private func handleTransportError(_ error: Error) {
        pingTimer?.invalidate()
        pingTimer = nil
        task = nil
        state = .offlineFallback
        // Soft reconnect with backoff.
        reconnectAttempt += 1
        let delay = min(8.0, pow(1.6, Double(min(reconnectAttempt, 6))))
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self, self.preferServer else { return }
            self.connect()
        }
        _ = error
    }

    private func schedulePing() {
        pingTimer?.invalidate()
        pingTimer = Timer.scheduledTimer(withTimeInterval: 20, repeats: true) { [weak self] _ in
            self?.task?.sendPing { _ in }
        }
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

