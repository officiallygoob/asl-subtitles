import Foundation
import SwiftUI

/// Scaffold for 1:1 in-app video calling (WebRTC / LiveKit next).
@MainActor
final class InAppCallController: ObservableObject {
    enum Phase: String {
        case idle, creatingLink, waitingForPeer, connecting, connected, ended
    }

    @Published var phase: Phase = .idle
    @Published var inviteLink: String = ""
    @Published var joinField: String = ""
    @Published var statusMessage: String = "Private video call with live ASL captions on your friend’s video"
    @Published var isMicOn: Bool = true
    @Published var isCameraOn: Bool = true
    @Published var remoteName: String = "Friend"
    @Published var lastError: String?

    private(set) var roomID: String = ""

    func createInvite() {
        phase = .creatingLink
        roomID = String(UUID().uuidString.prefix(8)).lowercased()
        inviteLink = "aslsubtitles://call/\(roomID)"
        phase = .waitingForPeer
        statusMessage = "Share this link. When they join, video connects and we caption their signing."
    }

    func joinWithLink() {
        let trimmed = joinField.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        if let id = trimmed.split(separator: "/").last, !id.isEmpty {
            roomID = String(id)
        }
        phase = .connecting
        statusMessage = "Connecting… (signaling stub — wire WebRTC/LiveKit next)"
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.85) { [weak self] in
            guard let self else { return }
            self.phase = .connected
            self.statusMessage = "Connected (UI preview). Replace placeholders with real remote frames."
        }
    }

    func endCall() {
        phase = .ended
        statusMessage = "Call ended"
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) { [weak self] in
            self?.phase = .idle
            self?.inviteLink = ""
            self?.joinField = ""
            self?.statusMessage = "Private video call with live ASL captions on your friend’s video"
        }
    }

    func toggleMic() { isMicOn.toggle() }
    func toggleCamera() { isCameraOn.toggle() }
}
