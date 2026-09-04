import Foundation

enum ConversationRole: String, Codable {
    case signing   // them → you (sign language → English)
    case speaking  // you → them (speech → text)
    case system
}

struct ConversationTurn: Identifiable, Equatable, Codable {
    var id: UUID = UUID()
    var role: ConversationRole
    var text: String
    var gloss: [String] = []
    var confidence: Double = 1
    var timestamp: Date = Date()
    var isPartial: Bool = false
}
