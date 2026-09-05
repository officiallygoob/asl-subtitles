import AppIntents
import Foundation

/// Spotlight / Shortcuts-visible past transcript (on-device only).
struct ConversationEntity: AppEntity, Identifiable {
    static var typeDisplayRepresentation = TypeDisplayRepresentation(name: "ASL Conversation")
    static var defaultQuery = ConversationEntityQuery()

    var id: UUID
    var title: String
    var preview: String
    var updatedAt: Date

    var displayRepresentation: DisplayRepresentation {
        DisplayRepresentation(title: "\(title)", subtitle: "\(preview)")
    }
}

struct ConversationEntityQuery: EntityQuery {
    func entities(for identifiers: [UUID]) async throws -> [ConversationEntity] {
        let all = await MainActor.run { ConversationStore.shared.conversations }
        return all.filter { identifiers.contains($0.id) }.map {
            ConversationEntity(id: $0.id, title: $0.title, preview: $0.preview, updatedAt: $0.updatedAt)
        }
    }

    func suggestedEntities() async throws -> [ConversationEntity] {
        let all = await MainActor.run { ConversationStore.shared.conversations }
        return all.prefix(12).map {
            ConversationEntity(id: $0.id, title: $0.title, preview: $0.preview, updatedAt: $0.updatedAt)
        }
    }
}

extension ConversationEntityQuery: EntityStringQuery {
    func entities(matching string: String) async throws -> [ConversationEntity] {
        let q = string.lowercased()
        let all = await MainActor.run { ConversationStore.shared.conversations }
        return all.filter {
            $0.title.lowercased().contains(q)
                || $0.preview.lowercased().contains(q)
                || $0.signingTranscript.lowercased().contains(q)
        }.map {
            ConversationEntity(id: $0.id, title: $0.title, preview: $0.preview, updatedAt: $0.updatedAt)
        }
    }
}

struct OpenPastConversationIntent: AppIntent {
    static var title: LocalizedStringResource = "Open Past ASL Conversation"
    static var description = IntentDescription("Open a saved ASL Subtitles transcript.")
    static var openAppWhenRun: Bool = true

    @Parameter(title: "Conversation")
    var conversation: ConversationEntity?

    func perform() async throws -> some IntentResult & ProvidesDialog {
        guard let conversation else {
            return .result(dialog: "Pick a saved conversation.")
        }
        await MainActor.run {
            ASLAppBridge.session?.openPersistedConversation(id: conversation.id)
        }
        return .result(dialog: IntentDialog(stringLiteral: "Opened \(conversation.title)"))
    }
}
