import Foundation

/// Offline gloss → English with non-manual marker (NMM) conditioning.
/// Mirrors server `gloss_english.py` rules at a smaller vocabulary scale.
enum GlossEnglish {
    /// Map a single gloss / heuristic label to a short English phrase.
    static func phrase(for gloss: String) -> String {
        let g = normalize(gloss)
        if g.count == 1, g.rangeOfCharacter(from: .letters) != nil {
            return g
        }
        return singleGloss[g] ?? g.split(separator: "-").map { $0.capitalized }.joined(separator: " ")
    }

    /// Apply NMM rules to English derived from a gloss label.
    /// - raised brows → question phrasing
    /// - head shake / negative face → negation
    /// - lean / shoulder tilt → mild emphasis
    static func english(gloss: String, nmm: NMMState) -> String {
        var text = phrase(for: gloss)
        guard !text.isEmpty else { return text }

        text = applyNegation(text, gloss: normalize(gloss), nmm: nmm)
        text = applyQuestion(text, gloss: normalize(gloss), nmm: nmm)
        text = applyEmphasis(text, nmm: nmm)
        return finalize(text)
    }

    /// Apply NMM conditioning to an already-fluent English string (server / multi-gloss).
    static func condition(english: String, glosses: [String] = [], nmm: NMMState) -> String {
        var text = english.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return text }
        let primary = glosses.first.map(normalize) ?? ""
        text = applyNegation(text, gloss: primary, nmm: nmm)
        text = applyQuestion(text, gloss: primary, nmm: nmm)
        text = applyEmphasis(text, nmm: nmm)
        return finalize(text)
    }

    // MARK: - Rules

    private static func applyQuestion(_ text: String, gloss: String, nmm: NMMState) -> String {
        guard nmm.isQuestionLikely else { return text }
        let wh = ["WHAT", "WHERE", "WHEN", "WHO", "WHY", "HOW", "WHICH"]
        if wh.contains(gloss) {
            return text.hasSuffix("?") ? text : text.trimmingCharacters(in: CharacterSet(charactersIn: ".")) + "?"
        }
        // Already a question phrase
        if text.hasSuffix("?") { return text }
        let stripped = text.trimmingCharacters(in: CharacterSet(charactersIn: ".!"))
        let lower = stripped.lowercased()
        // Avoid double-wrapping
        if lower.hasPrefix("are you") || lower.hasPrefix("do you") || lower.hasPrefix("what")
            || lower.hasPrefix("where") || lower.hasPrefix("how") || lower.hasPrefix("who")
            || lower.hasPrefix("why") || lower.hasPrefix("when") || lower.hasPrefix("which") {
            return stripped + "?"
        }
        // Statement → soft yes/no question
        if lower.hasPrefix("i ") || lower.hasPrefix("i'm ") {
            let rest = stripped.dropFirst(2).trimmingCharacters(in: .whitespaces)
            if lower.hasPrefix("i'm ") {
                return "Are you " + stripped.dropFirst(4).trimmingCharacters(in: .whitespaces) + "?"
            }
            return "Do you " + rest.lowercased() + "?"
        }
        if lower == "you" || lower.hasPrefix("you ") {
            return "Are you " + (lower == "you" ? "okay" : String(stripped.dropFirst(4))) + "?"
        }
        // Generic: append ?
        if stripped.count <= 24 {
            return "Are you " + lower + "?"
        }
        return stripped + "?"
    }

    private static func applyNegation(_ text: String, gloss: String, nmm: NMMState) -> String {
        guard nmm.isNegationLikely else { return text }
        if gloss == "NO" || gloss == "DONT-KNOW" { return text }
        let lower = text.lowercased()
        if lower.hasPrefix("not ") || lower.hasPrefix("no ") || lower.contains("don't") || lower.contains("n’t") {
            return text
        }
        let stripped = text.trimmingCharacters(in: CharacterSet(charactersIn: ".!?"))
        if stripped.lowercased().hasPrefix("i ") {
            let rest = stripped.dropFirst(2).trimmingCharacters(in: .whitespaces)
            return "I don't " + rest.lowercased()
        }
        if stripped.lowercased().hasPrefix("i'm ") {
            return "I'm not " + stripped.dropFirst(4).trimmingCharacters(in: .whitespaces).lowercased()
        }
        return "Not " + stripped.lowercased()
    }

    private static func applyEmphasis(_ text: String, nmm: NMMState) -> String {
        guard nmm.isEmphasisLikely else { return text }
        // Mild emphasis — avoid fake mouth-morpheme linguistics.
        let stripped = text.trimmingCharacters(in: CharacterSet(charactersIn: ".!?"))
        let punct: String
        if text.hasSuffix("?") { punct = "?" }
        else if text.hasSuffix("!") { punct = "!" }
        else { punct = "." }
        // Don't shout; a trailing emphasis marker via punctuation only when not a question.
        if punct == "?" { return stripped + "?" }
        return stripped + "!"
    }

    private static func finalize(_ text: String) -> String {
        var t = text.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return t }
        // Capitalize first letter
        let first = t.prefix(1).uppercased()
        t = first + t.dropFirst()
        return t
    }

    private static func normalize(_ gloss: String) -> String {
        gloss.uppercased()
            .replacingOccurrences(of: " ", with: "-")
            .filter { $0.isLetter || $0 == "-" }
    }

    private static let singleGloss: [String: String] = [
        "HELLO": "Hello.",
        "HI": "Hi.",
        "BYE": "Bye.",
        "THANKS": "Thank you.",
        "PLEASE": "Please.",
        "YES": "Yes.",
        "NO": "No.",
        "HELP": "Help.",
        "SORRY": "Sorry.",
        "EXCUSE": "Excuse me.",
        "ME": "Me.",
        "YOU": "You.",
        "WE": "We.",
        "MY": "My.",
        "YOUR": "Your.",
        "NAME": "Name.",
        "FRIEND": "Friend.",
        "FAMILY": "Family.",
        "WHAT": "What?",
        "WHERE": "Where?",
        "WHEN": "When?",
        "WHO": "Who?",
        "WHY": "Why?",
        "HOW": "How?",
        "WHICH": "Which?",
        "GOOD": "Good.",
        "BAD": "Bad.",
        "FINE": "Fine.",
        "GREAT": "Great.",
        "MORE": "More.",
        "WANT": "Want.",
        "NEED": "Need.",
        "LIKE": "Like.",
        "LOVE": "Love.",
        "GO": "Go.",
        "COME": "Come.",
        "STOP": "Stop.",
        "WAIT": "Wait.",
        "UNDERSTAND": "I understand.",
        "KNOW": "I know.",
        "DONT-KNOW": "I don't know.",
        "EAT": "Eat.",
        "DRINK": "Drink.",
        "HOME": "Home.",
        "WORK": "Work.",
        "SCHOOL": "School.",
        "TODAY": "Today.",
        "TOMORROW": "Tomorrow.",
        "HUNGRY": "Hungry.",
        "TIRED": "Tired.",
        "HAPPY": "Happy.",
        "SAD": "Sad.",
        "HOT": "Hot.",
        "COLD": "Cold.",
        "OK": "OK.",
        "MAYBE": "Maybe.",
        "AGAIN": "Again.",
        "SLOW": "Slow please.",
        "LOOK": "Look.",
        "SPELL": "Please spell that.",
        "WRITE": "Write.",
        "TRUE": "True.",
        "FALSE": "False.",
        "SAME": "Same.",
        "DIFFERENT": "Different.",
        "GOOD-MORNING": "Good morning.",
        "GOOD-NIGHT": "Good night.",
        "QUESTION": "Question.",
        "WATER": "Water.",
        "FOOD": "Food.",
        "BATHROOM": "Bathroom.",
        "PHONE": "Phone."
    ]
}
