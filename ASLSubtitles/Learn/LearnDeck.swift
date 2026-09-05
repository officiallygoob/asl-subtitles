import Foundation

struct LearnCard: Identifiable, Hashable {
    var id: String { gloss }
    /// Recognition gloss (uppercase, matches heuristics / server).
    var gloss: String
    /// Display title.
    var title: String
    var hint: String
    var symbolName: String
    var isLetter: Bool
}

enum LearnDeck {
    /// Fingerspelling A–Z + heuristic-supported everyday vocab.
    static var allCards: [LearnCard] {
        var cards: [LearnCard] = []
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" {
            let s = String(ch)
            cards.append(
                LearnCard(
                    gloss: s,
                    title: s,
                    hint: "Fingerspell \(s). Hold steady in frame.",
                    symbolName: "hand.raised.fill",
                    isLetter: true
                )
            )
        }
        for entry in VocabularyCatalog.entries where entry.heuristicSupported && entry.category != .fingerspelling {
            let gloss = entry.word.uppercased().replacingOccurrences(of: " ", with: "-")
            cards.append(
                LearnCard(
                    gloss: gloss,
                    title: entry.word,
                    hint: entry.tip,
                    symbolName: symbol(for: entry.category),
                    isLetter: false
                )
            )
        }
        return cards
    }

    static func orderedQueue(progress: [String: LearnCardProgress]) -> [LearnCard] {
        let now = Date()
        return allCards.sorted { a, b in
            let pa = progress[a.gloss]
            let pb = progress[b.gloss]
            let da = pa?.dueAt ?? .distantPast
            let db = pb?.dueAt ?? .distantPast
            if da != db { return da < db }
            // Prefer unseen, then letters first for onboarding.
            let ua = pa == nil
            let ub = pb == nil
            if ua != ub { return ua && !ub }
            if a.isLetter != b.isLetter { return a.isLetter && !b.isLetter }
            return a.gloss < b.gloss
        }.filter { card in
            // Always include due / unseen; lightly include a few future for short sessions
            guard let p = progress[card.gloss] else { return true }
            return p.dueAt <= now.addingTimeInterval(3600)
        }
    }

    private static func symbol(for category: VocabularyEntry.Category) -> String {
        switch category {
        case .greeting: return "hand.wave.fill"
        case .courtesy: return "heart.fill"
        case .people: return "person.fill"
        case .questions: return "questionmark.circle.fill"
        case .answers: return "checkmark.circle.fill"
        case .feelings: return "face.smiling.fill"
        case .food: return "fork.knife"
        case .time: return "clock.fill"
        case .places: return "mappin.circle.fill"
        case .verbs: return "figure.walk"
        case .numbers: return "number"
        case .social: return "bubble.left.and.bubble.right.fill"
        case .everyday: return "star.fill"
        case .fingerspelling: return "hand.raised.fill"
        case .needsML: return "brain.head.profile"
        }
    }
}
