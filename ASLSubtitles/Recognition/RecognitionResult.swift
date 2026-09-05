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
    /// When false, offline heuristics cannot approximate this gloss; needs ML / friend data.
    let heuristicSupported: Bool

    init(word: String, category: Category, tip: String, heuristicSupported: Bool = true) {
        self.word = word
        self.category = category
        self.tip = tip
        self.heuristicSupported = heuristicSupported
    }

    enum Category: String, CaseIterable {
        case fingerspelling = "Fingerspelling"
        case greeting = "Greetings"
        case courtesy = "Courtesy"
        case people = "People / Pronouns"
        case questions = "Questions"
        case answers = "Answers"
        case feelings = "Feelings"
        case food = "Food & Drink"
        case time = "Time"
        case places = "Places"
        case verbs = "Common Verbs"
        case numbers = "Numbers"
        case social = "Social Phrases"
        case everyday = "Everyday"
        case needsML = "Needs ML / Friend Data"
    }
}
