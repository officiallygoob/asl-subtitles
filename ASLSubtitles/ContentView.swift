import SwiftUI

/// Root view — Conversation Mode is the default experience.
struct ContentView: View {
    @StateObject private var session = ASLSessionController()

    var body: some View {
        ConversationView(session: session)
    }
}

#Preview {
    ContentView()
}
