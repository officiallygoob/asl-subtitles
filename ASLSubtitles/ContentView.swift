import SwiftUI

/// Root tabs: In Person | Call | Learn | History.
struct ContentView: View {
    @StateObject private var session = ASLSessionController()
    @State private var tab: AppTab = .inPerson

    enum AppTab: Hashable {
        case inPerson, call, learn, history
    }

    var body: some View {
        TabView(selection: $tab) {
            ConversationView(session: session)
                .tabItem { Label("In Person", systemImage: "person.2.fill") }
                .tag(AppTab.inPerson)

            CallTabView(session: session)
                .tabItem { Label("Call", systemImage: "video.fill") }
                .tag(AppTab.call)

            LearnTabView(session: session)
                .tabItem { Label("Learn", systemImage: "rectangle.on.rectangle.angled") }
                .tag(AppTab.learn)

            NavigationStack {
                HistorySheet(session: session)
            }
            .tabItem { Label("History", systemImage: "clock.arrow.circlepath") }
            .tag(AppTab.history)
        }
        .tint(AppTheme.accent)
        .preferredColorScheme(.dark)
        .onAppear { session.prepare() }
    }
}

#Preview {
    ContentView()
}
