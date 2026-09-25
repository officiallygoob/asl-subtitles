import Foundation

/// Routes `aslsubtitles://call/<roomID>?from=<username>` into the Call tab.
@MainActor
final class CallDeepLinkCenter: ObservableObject {
    @Published var pendingInvite: CallInvite?
    /// Bumps on every accepted link so the tab switches even if Call is already selected.
    @Published var focusToken: Int = 0

    func handle(_ url: URL) {
        guard let invite = CallInviteLink.parse(url: url) else { return }
        pendingInvite = invite
        focusToken &+= 1
    }
}
