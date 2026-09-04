import Foundation

enum RecognitionKind: String, Equatable {
    case fingerspell
    case everydaySign
    case unknown
}

struct RecognitionResult: Equatable {
    let label: String
    let kind: RecognitionKind
    let confidence: Double
    let timestamp: Date

    static let empty = RecognitionResult(
        label: "",
        kind: .unknown,
        confidence: 0,
        timestamp: .distantPast
    )

    var isConfident: Bool { confidence >= 0.55 && !label.isEmpty }
}

struct VocabularyEntry: Identifiable, Hashable {
    let id = UUID()
    let word: String
    let category: Category
    let tip: String

    enum Category: String, CaseIterable {
        case fingerspelling = "Fingerspelling"
        case greeting = "Greetings"
        case courtesy = "Courtesy"
        case people = "People"
        case everyday = "Everyday"
    }
}
