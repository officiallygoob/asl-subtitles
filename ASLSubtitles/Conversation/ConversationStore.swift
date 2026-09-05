import Foundation

/// On-device persistence for past conversation transcripts (clearable, never uploaded).
struct PersistedConversation: Codable, Identifiable, Hashable {
    var id: UUID
    var createdAt: Date
    var updatedAt: Date
    var title: String
    var turns: [PersistedTurn]
    var signingTranscript: String

    struct PersistedTurn: Codable, Hashable {
        var role: String
        var text: String
        var gloss: [String]
        var confidence: Double
        var createdAt: Date
    }

    var preview: String {
        let body = signingTranscript.isEmpty
            ? turns.map(\.text).joined(separator: " · ")
            : signingTranscript
        return String(body.prefix(120))
    }
}

@MainActor
final class ConversationStore: ObservableObject {
    static let shared = ConversationStore()

    @Published private(set) var conversations: [PersistedConversation] = []

    private let fileName = "conversation_history.json"
    private var fileURL: URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        let folder = dir.appendingPathComponent("ASLSubtitles", isDirectory: true)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder.appendingPathComponent(fileName)
    }

    init() {
        load()
    }

    func load() {
        guard let data = try? Data(contentsOf: fileURL),
              let decoded = try? JSONDecoder().decode([PersistedConversation].self, from: data) else {
            conversations = []
            return
        }
        conversations = decoded.sorted { $0.updatedAt > $1.updatedAt }
    }

    func save(_ conversation: PersistedConversation) {
        var next = conversations.filter { $0.id != conversation.id }
        next.insert(conversation, at: 0)
        // Cap retained history
        if next.count > 40 { next = Array(next.prefix(40)) }
        conversations = next
        persist()
    }

    func upsertLive(
        id: UUID,
        turns: [ConversationTurn],
        signingTranscript: String
    ) {
        let mapped = turns.map {
            PersistedConversation.PersistedTurn(
                role: $0.role == .signing ? "Signing" : ($0.role == .speaking ? "You said" : "System"),
                text: $0.text,
                gloss: $0.gloss,
                confidence: $0.confidence,
                createdAt: $0.timestamp
            )
        }
        guard !mapped.isEmpty || !signingTranscript.isEmpty else { return }
        let title: String = {
            if let first = mapped.first(where: { !$0.text.isEmpty }) {
                return String(first.text.prefix(42))
            }
            return "Conversation"
        }()
        let existing = conversations.first(where: { $0.id == id })
        let record = PersistedConversation(
            id: id,
            createdAt: existing?.createdAt ?? Date(),
            updatedAt: Date(),
            title: title,
            turns: mapped,
            signingTranscript: signingTranscript
        )
        save(record)
    }

    func delete(id: UUID) {
        conversations.removeAll { $0.id == id }
        persist()
    }

    func clearAll() {
        conversations = []
        persist()
    }

    private func persist() {
        guard let data = try? JSONEncoder().encode(conversations) else { return }
        try? data.write(to: fileURL, options: [.atomic])
    }
}
