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

    /// Local display username (not a phone number).
    @Published var username: String = UserDefaults.standard.string(forKey: "asl.username") ?? "" {
        didSet {
            let cleaned = Self.sanitizeUsername(username)
            if cleaned != username { username = cleaned; return }
            UserDefaults.standard.set(cleaned, forKey: "asl.username")
        }
    }

    private(set) var roomID: String = ""

    var displayName: String {
        let u = username.trimmingCharacters(in: .whitespacesAndNewlines)
        return u.isEmpty ? "You" : u
    }

    func createInvite() {
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

    /// Letters, numbers, underscore; 3–24 chars; no phone-like strings.
    static func sanitizeUsername(_ raw: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_"))
        let filtered = String(raw.unicodeScalars.filter { allowed.contains($0) })
        return String(filtered.prefix(24))
    }
}
