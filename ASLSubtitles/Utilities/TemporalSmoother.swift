import Foundation

/// Majority-vote temporal smoother to keep live words from flickering.
/// Persistence of the conversation transcript lives in ASLSessionController —
/// this only stabilizes the *current* word and can fade it after a short hold.
final class TemporalSmoother {
    private var window: [(label: String, confidence: Double)] = []
    private let windowSize: Int
    private let minVotes: Int
    private let holdDuration: TimeInterval

    private(set) var currentLabel: String = ""
    private(set) var currentConfidence: Double = 0
    private var lastAcceptedAt: Date = .distantPast

    init(windowSize: Int = 8, minVotes: Int = 4, holdDuration: TimeInterval = 1.1) {
        self.windowSize = windowSize
        self.minVotes = minVotes
        self.holdDuration = holdDuration
    }

    func reset() {
        window.removeAll()
        currentLabel = ""
        currentConfidence = 0
        lastAcceptedAt = .distantPast
    }

    /// Push a raw recognition; returns display label, confidence, empty flag,
    /// and whether a *new* non-empty label was just accepted (for offline finals).
    @discardableResult
    func push(_ result: RecognitionResult) -> (text: String, confidence: Double, isEmpty: Bool, isNewAcceptance: Bool) {
        if result.label.isEmpty || result.confidence < 0.45 {
            window.append(("", 0))
        } else {
            window.append((result.label, result.confidence))
        }
        if window.count > windowSize {
            window.removeFirst(window.count - windowSize)
        }

        let meaningful = window.filter { !$0.label.isEmpty }
        let counts = Dictionary(grouping: meaningful, by: \.label).mapValues(\.count)
        let best = counts.max(by: { $0.value < $1.value })

        var isNewAcceptance = false
        if let best, best.value >= minVotes {
            let confs = meaningful.filter { $0.label == best.key }.map(\.confidence)
            let avg = confs.reduce(0, +) / Double(max(confs.count, 1))
            let display = displayForm(best.key)
            if display != currentLabel && !display.isEmpty {
                isNewAcceptance = true
            }
            currentLabel = display
            currentConfidence = avg
            lastAcceptedAt = Date()
        } else if Date().timeIntervalSince(lastAcceptedAt) > holdDuration {
            currentConfidence = max(0, currentConfidence - 0.08)
            if currentConfidence < 0.25 {
                currentLabel = ""
                currentConfidence = 0
            }
        }

        let watching = currentLabel.isEmpty
        return (currentLabel, currentConfidence, watching, isNewAcceptance)
    }
}

private func displayForm(_ raw: String) -> String {
    // Preserve NMM-conditioned English phrases (spaces / punctuation).
    if raw.contains(" ") || raw.contains("?") || raw.contains("!") || raw.contains("'") {
        return raw
    }
    if raw.count == 1, raw.rangeOfCharacter(from: .letters) != nil {
        return raw.uppercased()
    }
    return raw.capitalized
}
