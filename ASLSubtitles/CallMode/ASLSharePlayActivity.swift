import GroupActivities
import Foundation

/// SharePlay activity: bring ASL Subtitles into a FaceTime call as a custom Group Activity.
struct ASLSubtitlesActivity: GroupActivity {
    static let activityIdentifier = "com.officiallygoob.aslsubtitles.conversation"

    var metadata: GroupActivityMetadata {
        var meta = GroupActivityMetadata()
        meta.title = "ASL Subtitles"
        meta.subtitle = "Live signing captions on this FaceTime call"
        meta.type = .generic
        #if os(iOS)
        meta.supportsContinuationOnTV = false
        #endif
        return meta
    }
}

@MainActor
final class SharePlayCoordinator: ObservableObject {
    @Published var isSessionActive = false
    @Published var status: String = ""

    private var session: GroupSession<ASLSubtitlesActivity>?
    private var messenger: GroupSessionMessenger?

    func startSharing() async {
        let activity = ASLSubtitlesActivity()
        switch await activity.prepareForActivation() {
        case .activationPreferred, .activationDisabled:
            do {
                _ = try await activity.activate()
                status = "SharePlay offered on FaceTime"
            } catch {
                status = "SharePlay unavailable: \(error.localizedDescription)"
            }
        default:
            status = "SharePlay not preferred right now"
        }
    }

    func observeSessions() {
        Task {
            for await session in ASLSubtitlesActivity.sessions() {
                self.session = session
                self.messenger = GroupSessionMessenger(session: session)
                session.join()
                self.isSessionActive = true
                self.status = "SharePlay session joined"
                Task {
                    for await state in session.$state.values {
                        if case .invalidated = state {
                            self.isSessionActive = false
                            self.status = "SharePlay ended"
                        }
                    }
                }
            }
        }
    }

    /// Optional lightweight sync of latest caption string to other participants.
    func broadcastCaption(_ text: String) {
        guard let messenger else { return }
        Task {
            try? await messenger.send(text)
        }
    }
}
