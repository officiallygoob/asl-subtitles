import Foundation

/// On-device Apple Intelligence helpers with graceful fallback when
/// Foundation Models / Apple Intelligence is unavailable (device/region/OS).
enum AppleIntelligenceText {
    static var isAvailable: Bool {
        #if canImport(FoundationModels)
        if #available(iOS 27.0, *) { return true }
        #endif
        return false
    }

    /// Gloss / rough caption → fluent English. Falls back to input on failure.
    static func polishEnglish(_ raw: String) async -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return trimmed }
        if let out = await generateOnDevice(
            instructions: "Polish short ASL-caption English into one natural sentence. Keep meaning, person, question/negation. Reply with only the sentence.",
            prompt: trimmed
        ) {
            return out
        }
        return trimmed
    }

    /// Conversation turns → short paragraph summary.
    static func summarizeConversation(_ turns: [(role: String, text: String)]) async -> String {
        guard !turns.isEmpty else { return "" }
        let blob = turns.map { "\($0.role): \($0.text)" }.joined(separator: "\n")
        if let out = await generateOnDevice(
            instructions: "Summarize this ASL/spoken conversation in 1–3 short sentences. Plain and accurate. Reply with only the summary.",
            prompt: blob
        ) {
            return out
        }
        let signing = turns.filter { $0.role.lowercased().contains("sign") }.suffix(4).map(\.text)
        let speaking = turns.filter {
            $0.role.lowercased().contains("you") || $0.role.lowercased().contains("speak")
        }.suffix(3).map(\.text)
        var parts: [String] = []
        if !signing.isEmpty { parts.append("Signed: " + signing.joined(separator: " · ")) }
        if !speaking.isEmpty { parts.append("Said: " + speaking.joined(separator: " · ")) }
        return parts.joined(separator: " ")
    }

    private static func generateOnDevice(instructions: String, prompt: String) async -> String? {
        #if canImport(FoundationModels)
        if #available(iOS 27.0, *) {
            return await FoundationModelsBridge.generate(instructions: instructions, prompt: prompt)
        }
        #endif
        return nil
    }
}

#if canImport(FoundationModels)
import FoundationModels

@available(iOS 27.0, *)
enum FoundationModelsBridge {
    static func generate(instructions: String, prompt: String) async -> String? {
        // TODO(iOS27-beta): pin exact API once Xcode beta stabilizes.
        // Uses LanguageModelSession when present; soft-fails otherwise.
        do {
            let session = LanguageModelSession {
                instructions
            }
            let response = try await session.respond(to: prompt)
            let text = "\(response.content)".trimmingCharacters(in: .whitespacesAndNewlines)
            return text.isEmpty ? nil : text
        } catch {
            return nil
        }
    }
}
#endif
