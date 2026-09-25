import SwiftUI

@main
struct ASLSubtitlesApp: App {
    @StateObject private var callLinks = CallDeepLinkCenter()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(callLinks)
                .onOpenURL { callLinks.handle($0) }
        }
    }
}
