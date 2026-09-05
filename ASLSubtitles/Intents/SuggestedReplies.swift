import Foundation

enum SuggestedReplies {
    /// 2–3 short things a hearing user might say next.
    static func suggest(afterSigning english: String, history: [(role: String, text: String)] = []) async -> [String] {
        let trimmed = english.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        #if canImport(FoundationModels)
        if #available(iOS 27.0, *) {
            let polished = await AppleIntelligenceText.polishEnglish(
                """
                Suggest 3 very short spoken replies (under 8 words each) a hearing person could say next \
                after their deaf friend signed: \"\(trimmed)\". Reply as three lines only, no numbering.
                """
            )
            let lines = polished
                .split(whereSeparator: \.isNewline)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .map { $0.trimmingCharacters(in: CharacterSet(charactersIn: "-•*0123456789. ")) }
                .filter { !$0.isEmpty && $0.count < 48 }
            if lines.count >= 2 {
                return Array(lines.prefix(3))
            }
        }
        #endif

        return templateReplies(for: trimmed)
    }

    static func templateReplies(for english: String) -> [String] {
        let low = english.lowercased()
        if low.contains("?") || low.hasPrefix("are you") || low.hasPrefix("do you") || low.hasPrefix("what") || low.hasPrefix("how") {
            return ["Yes", "Not sure yet", "Can you repeat?"]
        }
        if low.contains("hello") || low.hasPrefix("hi") {
            return ["Hi!", "How are you?", "Good to see you"]
        }
        if low.contains("thank") {
            return ["You're welcome", "Of course", "Anytime"]
        }
        if low.contains("help") || low.contains("need") {
            return ["I can help", "What do you need?", "One moment"]
        }
        if low.hasPrefix("not ") || low.hasPrefix("i don't") || low == "no." || low == "no" {
            return ["Okay", "Got it", "What works instead?"]
        }
        return ["Okay", "Thanks", "Tell me more"]
    }
}
