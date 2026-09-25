import AVFoundation
import Combine
import Foundation
import LiveKit

/// 1:1 in-app video call over LiveKit.
/// Identity: **usernames + invite links only** — no phone numbers / Contacts.
@MainActor
final class InAppCallController: ObservableObject {
    enum Phase: String {
        case idle, creatingLink, waitingForPeer, connecting, connected, ended
    }

    @Published var phase: Phase = .idle
    @Published var inviteLink: String = ""
    @Published var inviteCode: String = ""
    @Published var joinField: String = ""
    @Published var statusMessage: String = "Share a link or code — no phone numbers"
    @Published var isMicOn: Bool = true
    @Published var isCameraOn: Bool = true
    /// Front camera is the default so the other person sees you signing.
    @Published var prefersFrontCamera: Bool = true
    @Published var remoteName: String = "Friend"
    @Published var remoteCameraMuted: Bool = false
    @Published var lastError: String?
    /// User-facing validation message under the username field.
    @Published var usernameError: String?
    @Published var localVideoTrack: VideoTrack?
    @Published var remoteVideoTrack: VideoTrack?

    /// LAN token server. Not an identity and not a phone number.
    @Published var tokenServerURL: String {
        didSet { persistTokenServerURL() }
    }

    /// Local display username (not a phone number). Rejected names are not persisted.
    @Published var username: String = InAppCallController.loadPersistedUsername() {
        didSet {
            applyUsernameChange(username)
        }
    }

    /// Stop the in-person capture session before LiveKit opens the camera.
    var prepareForCallMedia: (() async -> Void)?
    /// Give the camera back to in-person recognition after the call releases it.
    var releaseCallMedia: (() async -> Void)?
    /// Remote landmarks from `RemoteSignFrameBridge` (on-device). Caption commit is a TODO.
    var onRemoteLandmarks: ((LandmarkFrame) -> Void)?

    private(set) var roomID: String = ""

    /// Last name that passed validation (kept in UserDefaults).
    private var persistedGoodUsername: String = InAppCallController.loadPersistedUsername()

    private let room = Room()
    private let roomEvents = CallRoomEvents()
    private let remoteSignBridge = RemoteSignFrameBridge()
    private var connectGeneration: UInt = 0
    /// Non-zero while this controller owns an active LiveKit session.
    private var liveSession: UInt = 0
    /// Disconnects we initiated. The SDK reports those on a later hop; don't treat them as drops.
    private var suppressedDisconnects = 0
    /// When true, end-of-call must not restart the in-person camera.
    private var callOwnsCamera = false
    private var mediaTail: Task<Void, Never> = Task {}

    static let tokenURLDefaultsKey = "asl.livekit.tokenURL"
    static let defaultTokenServerURL = "http://127.0.0.1:8787/v1/token"

    var displayName: String {
        let u = username.trimmingCharacters(in: .whitespacesAndNewlines)
        return u.isEmpty ? "You" : u
    }

    var showsLocalPreview: Bool {
        isInCallFlow
    }

    private var isInCallFlow: Bool {
        switch phase {
        case .creatingLink, .waitingForPeer, .connecting, .connected:
            return true
        case .idle, .ended:
            return false
        }
    }

    /// True when the current field value is a fully valid, non-empty username.
    var hasValidUsername: Bool {
        if case .ok = UsernameModerator.validate(username) { return true }
        return false
    }

    init() {
        tokenServerURL = Self.loadTokenServerURL()
        roomEvents.owner = self
        room.add(delegate: roomEvents)
        remoteSignBridge.onLandmarks = { [weak self] frame in
            Task { @MainActor in
                self?.onRemoteLandmarks?(frame)
            }
        }
    }

    func createInvite() {
        guard let name = validatedUsername() else { return }
        lastError = nil
        let id = String(UUID().uuidString.prefix(8)).lowercased()
        roomID = id
        inviteCode = id
        inviteLink = CallInviteLink.make(roomID: id, from: name)
        remoteName = "Friend"
        remoteCameraMuted = false
        phase = .connecting
        statusMessage = "Opening room \(inviteCode) as \(name)…"
        startConnect(roomID: id, username: name)
    }

    func joinWithLinkOrCode() {
        guard let name = validatedUsername() else { return }
        let trimmed = joinField.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        guard let invite = CallInviteLink.parse(text: trimmed) else {
            lastError = "Enter a call code or an aslsubtitles:// link."
            return
        }
        lastError = nil
        roomID = invite.roomID
        inviteCode = invite.roomID
        inviteLink = CallInviteLink.make(roomID: invite.roomID, from: name)
        if let from = invite.fromUsername {
            remoteName = from
        }
        remoteCameraMuted = false
        phase = .connecting
        let who = invite.fromUsername.map { " — \($0) invited you" } ?? ""
        statusMessage = "Joining \(invite.roomID) as \(name)\(who)…"
        startConnect(roomID: invite.roomID, username: name)
    }

    /// Deep link. Does not connect until the local username is valid.
    func applyIncomingInvite(_ invite: CallInvite) {
        joinField = invite.link
        if let from = invite.fromUsername {
            remoteName = from
        }
        guard validatedUsername() != nil else {
            statusMessage = "Choose a username to join room \(invite.roomID)."
            return
        }
        joinWithLinkOrCode()
    }

    func endCall() {
        connectGeneration &+= 1
        let endGeneration = connectGeneration
        liveSession = 0
        callOwnsCamera = false
        remoteSignBridge.detach()
        localVideoTrack = nil
        remoteVideoTrack = nil
        remoteCameraMuted = false
        phase = .ended
        statusMessage = "Call ended"
        Task {
            await self.disconnectRoom()
            guard self.connectGeneration == endGeneration else { return }
            await self.runMedia {
                guard !self.callOwnsCamera else { return }
                await self.releaseCallMedia?()
            }
            try? await Task.sleep(nanoseconds: 350_000_000)
            guard self.phase == .ended, self.connectGeneration == endGeneration else { return }
            self.phase = .idle
            self.inviteLink = ""
            self.inviteCode = ""
            self.joinField = ""
            self.remoteName = "Friend"
            self.localVideoTrack = nil
            self.remoteVideoTrack = nil
            self.statusMessage = "Share a link or code — no phone numbers"
        }
    }

    func toggleMic() {
        isMicOn.toggle()
        guard liveSession != 0 else { return }
        let enabled = isMicOn
        Task {
            do {
                try await self.room.localParticipant.setMicrophone(enabled: enabled)
            } catch {
                self.isMicOn = !enabled
                self.lastError = "Microphone: \(error.localizedDescription)"
            }
        }
    }

    func toggleCamera() {
        isCameraOn.toggle()
        guard liveSession != 0 else { return }
        let enabled = isCameraOn
        Task {
            do {
                if enabled {
                    let position: AVCaptureDevice.Position = self.prefersFrontCamera ? .front : .back
                    let publication = try await self.room.localParticipant.setCamera(
                        enabled: true,
                        captureOptions: CameraCaptureOptions(position: position)
                    )
                    if let track = publication?.track as? VideoTrack {
                        self.localVideoTrack = track
                    }
                } else {
                    try await self.room.localParticipant.setCamera(enabled: false)
                }
            } catch {
                self.isCameraOn = !enabled
                self.lastError = "Camera: \(error.localizedDescription)"
            }
        }
    }

    func flipCamera() {
        guard liveSession != 0 else { return }
        Task { await self.flipCameraNow() }
    }

    /// Letters, numbers, underscore; max 24 chars (format only — use `UsernameModerator.validate` for full rules).
    static func sanitizeUsername(_ raw: String) -> String {
        UsernameModerator.sanitizeFormat(raw)
    }

    // MARK: - Room events (main actor)

    func handleLocalPublished(_ publication: LocalTrackPublication) {
        guard isInCallFlow else { return }
        guard publication.source == .camera, let track = publication.track as? VideoTrack else { return }
        localVideoTrack = track
    }

    func handlePeerJoined(_ participant: RemoteParticipant) {
        guard phase == .connecting || phase == .waitingForPeer || phase == .connected else { return }
        remoteName = Self.label(for: participant)
        phase = .connected
        statusMessage = "\(remoteName) joined"
        attachRemoteCamera(from: participant)
    }

    func handlePeerLeft(_ participant: RemoteParticipant) {
        guard phase == .connected || phase == .waitingForPeer else { return }
        let gone = participant.identity?.stringValue
        let remaining = room.remoteParticipants.values.filter { $0.identity?.stringValue != gone }
        guard remaining.isEmpty else {
            if let next = remaining.first {
                remoteName = Self.label(for: next)
                attachRemoteCamera(from: next)
            }
            return
        }
        remoteSignBridge.detach()
        remoteVideoTrack = nil
        remoteCameraMuted = false
        remoteName = "Friend"
        phase = .waitingForPeer
        statusMessage = "Waiting for your friend to rejoin \(inviteCode)…"
    }

    func handleRemoteSubscribed(_ participant: RemoteParticipant, publication: RemoteTrackPublication) {
        guard isInCallFlow else { return }
        guard publication.kind == .video, publication.source != .screenShareVideo else { return }
        guard let track = publication.track as? VideoTrack else { return }
        if publication.source != .camera, remoteVideoTrack != nil { return }
        remoteName = Self.label(for: participant)
        remoteVideoTrack = track
        remoteCameraMuted = publication.isMuted
        if publication.isMuted {
            remoteSignBridge.detach()
        } else {
            remoteSignBridge.attach(track)
        }
        if phase == .connecting || phase == .waitingForPeer {
            phase = .connected
            statusMessage = "In a call with \(remoteName)"
        }
    }

    func handleMute(participant: Participant, publication: TrackPublication, isMuted: Bool) {
        guard isInCallFlow else { return }
        guard publication.source == .camera, participant is RemoteParticipant else { return }
        remoteCameraMuted = isMuted
        if isMuted {
            remoteSignBridge.detach()
        } else if let track = publication.track as? VideoTrack {
            remoteVideoTrack = track
            remoteSignBridge.attach(track)
        }
    }

    func handleUnexpectedDisconnect(_ error: LiveKitError?) {
        if suppressedDisconnects > 0 {
            suppressedDisconnects -= 1
            return
        }
        guard liveSession != 0 else { return }
        guard phase == .connected || phase == .waitingForPeer || phase == .connecting else { return }
        liveSession = 0
        callOwnsCamera = false
        connectGeneration &+= 1
        remoteSignBridge.detach()
        localVideoTrack = nil
        remoteVideoTrack = nil
        let detail = error?.localizedDescription ?? "The call disconnected."
        lastError = inviteCode.isEmpty ? detail : "Disconnected from \(inviteCode). \(detail)"
        phase = .idle
        statusMessage = lastError ?? detail
        Task {
            await self.runMedia {
                guard !self.callOwnsCamera else { return }
                await self.releaseCallMedia?()
            }
        }
    }

    func syncRemoteParticipants() {
        guard let participant = room.remoteParticipants.values.first else { return }
        remoteName = Self.label(for: participant)
        attachRemoteCamera(from: participant)
        phase = .connected
        statusMessage = "In a call with \(remoteName)"
    }

    // MARK: - Private

    private func startConnect(roomID: String, username: String) {
        connectGeneration &+= 1
        let generation = connectGeneration
        callOwnsCamera = true
        Task {
            await self.runConnect(generation: generation, roomID: roomID, username: username)
        }
    }

    private func runConnect(generation: UInt, roomID: String, username: String) async {
        await runMedia {
            guard self.callOwnsCamera else { return }
            await self.prepareForCallMedia?()
        }
        guard generation == connectGeneration else { return }

        do {
            liveSession = 0
            await disconnectRoom()
            guard generation == connectGeneration else { return }

            let tokenURL = try normalizedTokenURL()
            let details = try await LiveKitTokenClient.fetch(
                tokenURL: tokenURL,
                room: roomID,
                username: username
            )
            guard generation == connectGeneration else { return }
            guard details.room == roomID else {
                throw LiveKitTokenError.server("Token server returned a different room.")
            }

            try await room.connect(url: details.url, token: details.token)
            guard generation == connectGeneration else { return }

            await publishLocalMedia()
            guard generation == connectGeneration else { return }

            liveSession = generation
            if room.remoteParticipants.isEmpty {
                phase = .waitingForPeer
                statusMessage = "You're in the room as \(username). Share code \(inviteCode)."
            } else {
                syncRemoteParticipants()
            }
        } catch {
            guard generation == connectGeneration else { return }
            liveSession = 0
            callOwnsCamera = false
            await disconnectRoom()
            guard generation == connectGeneration else { return }
            await runMedia {
                guard !self.callOwnsCamera else { return }
                await self.releaseCallMedia?()
            }
            guard generation == connectGeneration else { return }
            let text = LiveKitTokenClient.message(for: error, tokenURL: tokenServerURL)
            lastError = text
            phase = .idle
            statusMessage = text
        }
    }

    private func publishLocalMedia() async {
        if isCameraOn {
            do {
                let position: AVCaptureDevice.Position = prefersFrontCamera ? .front : .back
                let publication = try await room.localParticipant.setCamera(
                    enabled: true,
                    captureOptions: CameraCaptureOptions(position: position)
                )
                if let track = publication?.track as? VideoTrack {
                    localVideoTrack = track
                }
            } catch {
                isCameraOn = false
                lastError = "Camera didn't start. \(error.localizedDescription)"
            }
        }
        if isMicOn {
            do {
                try await room.localParticipant.setMicrophone(enabled: true)
            } catch {
                isMicOn = false
                lastError = "Microphone didn't start. \(error.localizedDescription)"
            }
        }
    }

    private func flipCameraNow() async {
        guard let publication = room.localParticipant.localVideoTracks.first(where: { $0.source == .camera }),
              let local = publication.track as? LocalVideoTrack,
              let capturer = local.capturer as? CameraCapturer else {
            return
        }
        do {
            _ = try await capturer.switchCameraPosition()
            prefersFrontCamera.toggle()
        } catch {
            lastError = "Couldn't flip the camera. \(error.localizedDescription)"
        }
    }

    private func attachRemoteCamera(from participant: RemoteParticipant) {
        guard let publication = participant.videoTracks.first(where: { $0.source == .camera })
                ?? participant.videoTracks.first(where: { $0.source != .screenShareVideo }),
              let track = publication.track as? VideoTrack else {
            return
        }
        remoteVideoTrack = track
        remoteCameraMuted = publication.isMuted
        if publication.isMuted {
            remoteSignBridge.detach()
        } else {
            remoteSignBridge.attach(track)
        }
    }

    private func disconnectRoom() async {
        let state = room.connectionState
        // Already-idle disconnects return without a delegate callback.
        if state != .disconnected && state != .disconnecting {
            suppressedDisconnects += 1
        }
        await room.disconnect()
    }

    private func normalizedTokenURL() throws -> URL {
        let trimmed = tokenServerURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmed),
              let scheme = url.scheme?.lowercased(),
              scheme == "http" || scheme == "https",
              url.host != nil else {
            throw LiveKitTokenError.invalidURL
        }
        return url
    }

    private func runMedia(_ body: @escaping @MainActor () async -> Void) async {
        let previous = mediaTail
        let next = Task { @MainActor in
            _ = await previous.value
            await body()
        }
        mediaTail = next
        await next.value
    }

    private static func label(for participant: Participant) -> String {
        if let name = participant.name?.trimmingCharacters(in: .whitespacesAndNewlines), !name.isEmpty {
            return name
        }
        if let identity = participant.identity?.stringValue, !identity.isEmpty {
            return identity
        }
        return "Friend"
    }

    @discardableResult
    private func validatedUsername() -> String? {
        switch UsernameModerator.validate(username) {
        case .ok(let good):
            usernameError = nil
            return good
        case .rejected(let reason):
            usernameError = reason
            lastError = reason
            return nil
        }
    }

    private static func loadTokenServerURL() -> String {
        if let stored = UserDefaults.standard.string(forKey: tokenURLDefaultsKey)?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !stored.isEmpty {
            return stored
        }
        if let bundled = Bundle.main.object(forInfoDictionaryKey: "ASLLiveKitTokenURL") as? String {
            let trimmed = bundled.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { return trimmed }
        }
        return defaultTokenServerURL
    }

    private func persistTokenServerURL() {
        let trimmed = tokenServerURL.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            tokenServerURL = Self.defaultTokenServerURL
            return
        }
        if trimmed != tokenServerURL {
            tokenServerURL = trimmed
            return
        }
        UserDefaults.standard.set(trimmed, forKey: Self.tokenURLDefaultsKey)
    }

    private static func loadPersistedUsername() -> String {
        let raw = UserDefaults.standard.string(forKey: "asl.username") ?? ""
        if case .ok(let good) = UsernameModerator.validate(raw) {
            return good
        }
        // Drop legacy bad / incomplete persisted values.
        UserDefaults.standard.removeObject(forKey: "asl.username")
        return ""
    }

    private func applyUsernameChange(_ raw: String) {
        let cleaned = Self.sanitizeUsername(raw)
        if cleaned != username {
            username = cleaned
            return
        }

        switch UsernameModerator.validate(cleaned) {
        case .ok(let good):
            usernameError = nil
            persistedGoodUsername = good
            UserDefaults.standard.set(good, forKey: "asl.username")
        case .rejected(let reason):
            usernameError = cleaned.isEmpty ? nil : reason
            // Do not persist rejected names — keep previous good name or empty.
            if persistedGoodUsername.isEmpty {
                UserDefaults.standard.removeObject(forKey: "asl.username")
            } else {
                UserDefaults.standard.set(persistedGoodUsername, forKey: "asl.username")
            }
        }
    }
}

/// Forwards LiveKit callbacks onto `InAppCallController` (the SDK calls delegates off the main thread).
private final class CallRoomEvents: NSObject, RoomDelegate, @unchecked Sendable {
    nonisolated(unsafe) weak var owner: InAppCallController?

    func room(_ room: Room, participant: LocalParticipant, didPublishTrack publication: LocalTrackPublication) {
        let owner = owner
        Task { @MainActor in
            owner?.handleLocalPublished(publication)
        }
    }

    func room(_ room: Room, participantDidConnect participant: RemoteParticipant) {
        let owner = owner
        Task { @MainActor in
            owner?.handlePeerJoined(participant)
        }
    }

    func room(_ room: Room, participantDidDisconnect participant: RemoteParticipant) {
        let owner = owner
        Task { @MainActor in
            owner?.handlePeerLeft(participant)
        }
    }

    func room(_ room: Room, participant: RemoteParticipant, didSubscribeTrack publication: RemoteTrackPublication) {
        let owner = owner
        Task { @MainActor in
            owner?.handleRemoteSubscribed(participant, publication: publication)
        }
    }

    func room(
        _ room: Room,
        participant: Participant,
        trackPublication: TrackPublication,
        didUpdateIsMuted isMuted: Bool
    ) {
        let owner = owner
        Task { @MainActor in
            owner?.handleMute(participant: participant, publication: trackPublication, isMuted: isMuted)
        }
    }

    func room(_ room: Room, didDisconnectWithError error: LiveKitError?) {
        let owner = owner
        Task { @MainActor in
            owner?.handleUnexpectedDisconnect(error)
        }
    }
}
