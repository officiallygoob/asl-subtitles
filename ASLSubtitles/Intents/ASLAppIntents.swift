import AppIntents
import Foundation

/// Shared store so App Intents can poke the running session when present.
@MainActor
enum ASLAppBridge {
    static weak var session: ASLSessionController?

    static func bind(_ session: ASLSessionController) {
        self.session = session
    }
}

struct StartConversationIntent: AppIntent {
    static var title: LocalizedStringResource = "Start ASL Conversation"
    static var description = IntentDescription("Open Conversation Mode and start watching for signs.")
    static var openAppWhenRun: Bool = true

    func perform() async throws -> some IntentResult & ProvidesDialog {
        await MainActor.run {
            ASLAppBridge.session?.prepare()
            ASLAppBridge.session?.start()
        }
        return .result(dialog: "Starting ASL conversation.")
    }
}

struct ClearTranscriptIntent: AppIntent {
    static var title: LocalizedStringResource = "Clear ASL Captions"
    static var description = IntentDescription("Clear the signing transcript and conversation history.")
    static var openAppWhenRun: Bool = true

    func perform() async throws -> some IntentResult & ProvidesDialog {
        await MainActor.run {
            ASLAppBridge.session?.clearHistory()
        }
        return .result(dialog: "Cleared ASL captions.")
    }
}

struct ToggleMicIntent: AppIntent {
    static var title: LocalizedStringResource = "Toggle ASL Conversation Mic"
    static var description = IntentDescription("Turn speech-to-text listening on or off in Conversation Mode.")
    static var openAppWhenRun: Bool = true

    func perform() async throws -> some IntentResult & ProvidesDialog {
        let listening = await MainActor.run { () -> Bool in
            ASLAppBridge.session?.toggleSpeech()
            return ASLAppBridge.session?.speech.isListening ?? false
        }
        return .result(dialog: listening ? "Microphone on." : "Microphone off.")
    }
}

struct FlipCameraIntent: AppIntent {
    static var title: LocalizedStringResource = "Flip ASL Camera"
    static var description = IntentDescription("Switch between front and rear camera in ASL Subtitles.")
    static var openAppWhenRun: Bool = true

    func perform() async throws -> some IntentResult & ProvidesDialog {
        await MainActor.run {
            ASLAppBridge.session?.toggleCamera()
        }
        return .result(dialog: "Flipped the camera.")
    }
}

struct SummarizeConversationIntent: AppIntent {
    static var title: LocalizedStringResource = "Summarize ASL Conversation"
    static var description = IntentDescription("Summarize the current ASL Subtitles conversation with Apple Intelligence when available.")
    static var openAppWhenRun: Bool = true

    func perform() async throws -> some IntentResult & ProvidesDialog {
        let summary = await MainActor.run { () -> String in
            guard let session = ASLAppBridge.session else {
                return "Open ASL Subtitles first, then ask again."
            }
            return session.summarizeConversationSyncPlaceholder()
        }
        // Kick async polish; return immediate text for Siri.
        await MainActor.run {
            ASLAppBridge.session?.requestConversationSummary()
        }
        if summary.isEmpty {
            return .result(dialog: "Nothing to summarize yet.")
        }
        return .result(dialog: IntentDialog(stringLiteral: summary))
    }
}


struct StartCallModeIntent: AppIntent {
    static var title: LocalizedStringResource = "Start ASL Call Mode"
    static var description = IntentDescription("Use ASL Subtitles with FaceTime via screen capture / broadcast.")
    static var openAppWhenRun: Bool = true

    func perform() async throws -> some IntentResult & ProvidesDialog {
        await MainActor.run {
            ASLAppBridge.session?.startCallModeFromIntent()
        }
        return .result(dialog: "Starting Call Mode for FaceTime.")
    }
}


struct OpenLearnIntent: AppIntent {
    static var title: LocalizedStringResource = "Learn ASL"
    static var description = IntentDescription("Open the Learn ASL flashcard practice tab.")
    static var openAppWhenRun: Bool = true

    func perform() async throws -> some IntentResult & ProvidesDialog {
        return .result(dialog: "Open the Learn tab to practice signs.")
    }
}

struct ASLAppShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(
            intent: StartConversationIntent(),
            phrases: [
                "Start ASL conversation in \(.applicationName)",
                "Open ASL conversation in \(.applicationName)",
                "Start signing captions in \(.applicationName)"
            ],
            shortTitle: "Start conversation",
            systemImageName: "hands.and.sparkles.fill"
        )
        AppShortcut(
            intent: ClearTranscriptIntent(),
            phrases: [
                "Clear ASL captions in \(.applicationName)",
                "Clear signing transcript in \(.applicationName)"
            ],
            shortTitle: "Clear captions",
            systemImageName: "trash"
        )
        AppShortcut(
            intent: ToggleMicIntent(),
            phrases: [
                "Toggle mic in \(.applicationName)",
                "Toggle conversation microphone in \(.applicationName)"
            ],
            shortTitle: "Toggle mic",
            systemImageName: "mic.fill"
        )
        AppShortcut(
            intent: FlipCameraIntent(),
            phrases: [
                "Flip camera in \(.applicationName)",
                "Switch ASL camera in \(.applicationName)"
            ],
            shortTitle: "Flip camera",
            systemImageName: "arrow.triangle.2.circlepath.camera"
        )
        AppShortcut(
            intent: SummarizeConversationIntent(),
            phrases: [
                "Summarize ASL conversation in \(.applicationName)",
                "Summarize signing captions in \(.applicationName)"
            ],
            shortTitle: "Summarize",
            systemImageName: "text.justify.leading"
        )
        AppShortcut(
            intent: OpenLearnIntent(),
            phrases: [
                "Learn ASL in \(.applicationName)",
                "Practice ASL signs in \(.applicationName)"
            ],
            shortTitle: "Learn ASL",
            systemImageName: "rectangle.on.rectangle.angled"
        )
        AppShortcut(
            intent: StartCallModeIntent(),
            phrases: [
                "Start ASL Call Mode in \(.applicationName)",
                "Use \(.applicationName) with FaceTime",
                "Open Conversation Mode in \(.applicationName)"
            ],
            shortTitle: "Call Mode",
            systemImageName: "video.fill"
        )
        AppShortcut(
            intent: OpenPastConversationIntent(),
            phrases: [
                "Open past ASL conversation in \(.applicationName)",
                "Show ASL transcript history in \(.applicationName)"
            ],
            shortTitle: "Past chats",
            systemImageName: "clock.arrow.circlepath"
        )
    }
}
