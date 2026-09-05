import Foundation

enum RecognitionKind: String, Equatable {
    case fingerspell
    case everydaySign
    case unknown
}

struct RecognitionResult: Equatable {
    /// Display English (may already include NMM conditioning).
    let label: String
    let kind: RecognitionKind
    let confidence: Double
    let timestamp: Date
    /// Raw gloss token when available (e.g. HELLO).
    let gloss: String
    let nmm: NMMState?

    static let empty = RecognitionResult(
        label: "",
        kind: .unknown,
        confidence: 0,
        timestamp: .distantPast,
        gloss: "",
        nmm: nil
    )

    init(
        label: String,
        kind: RecognitionKind,
        confidence: Double,
        timestamp: Date = Date(),
        gloss: String = "",
        nmm: NMMState? = nil
    ) {
        self.label = label
        self.kind = kind
        self.confidence = confidence
        self.timestamp = timestamp
        self.gloss = gloss
        self.nmm = nmm
    }

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
