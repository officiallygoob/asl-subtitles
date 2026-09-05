import Foundation
import SwiftUI

/// Scaffold for 1:1 in-app video calling (WebRTC / LiveKit next).
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
    @Published var remoteName: String = "Friend"
    @Published var lastError: String?
    /// User-facing validation message under the username field.
    @Published var usernameError: String?

    /// Local display username (not a phone number). Rejected names are not persisted.
    @Published var username: String = InAppCallController.loadPersistedUsername() {
        didSet {
            applyUsernameChange(username)
        }
    }

    private(set) var roomID: String = ""

    /// Last name that passed validation (kept in UserDefaults).
    private var persistedGoodUsername: String = InAppCallController.loadPersistedUsername()

    var displayName: String {
        let u = username.trimmingCharacters(in: .whitespacesAndNewlines)
        return u.isEmpty ? "You" : u
    }

    /// True when the current field value is a fully valid, non-empty username.
    var hasValidUsername: Bool {
        if case .ok = UsernameModerator.validate(username) { return true }
        return false
    }

    func createInvite() {
        guard ensureValidUsernameForAction() else { return }
        phase = .creatingLink
        roomID = String(UUID().uuidString.prefix(8)).lowercased()
        inviteCode = roomID
        // Link + short code — usernames only in profile; never phone.
        let userHint = username.isEmpty ? "" : "?from=\(username)"
        inviteLink = "aslsubtitles://call/\(roomID)\(userHint)"
        phase = .waitingForPeer
        statusMessage = "Share the link or code \(inviteCode). They join with a username — no phone number needed."
    }

    func joinWithLinkOrCode() {
        guard ensureValidUsernameForAction() else { return }
        let trimmed = joinField.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        if let id = trimmed.split(separator: "/").last?
            .split(separator: "?").first,
           !id.isEmpty {
            roomID = String(id).lowercased()
        } else {
            roomID = trimmed.lowercased()
        }
        inviteCode = roomID
        phase = .connecting
        statusMessage = "Connecting as \(displayName)… (signaling stub — WebRTC/LiveKit next)"
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.85) { [weak self] in
            guard let self else { return }
            self.phase = .connected
            self.remoteName = "Friend"
            self.statusMessage = "Connected (UI preview). Wire WebRTC/LiveKit; identity stays username + link."
        }
    }

    func endCall() {
        phase = .ended
        statusMessage = "Call ended"
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) { [weak self] in
            self?.phase = .idle
            self?.inviteLink = ""
            self?.inviteCode = ""
            self?.joinField = ""
            self?.statusMessage = "Share a link or code — no phone numbers"
        }
    }

    func toggleMic() { isMicOn.toggle() }
    func toggleCamera() { isCameraOn.toggle() }

    /// Letters, numbers, underscore; max 24 chars (format only — use `UsernameModerator.validate` for full rules).
    static func sanitizeUsername(_ raw: String) -> String {
        UsernameModerator.sanitizeFormat(raw)
    }

    // MARK: - Private

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

    @discardableResult
    private func ensureValidUsernameForAction() -> Bool {
        switch UsernameModerator.validate(username) {
        case .ok:
            usernameError = nil
            return true
        case .rejected(let reason):
            usernameError = reason
            lastError = reason
            return false
        }
    }
}
