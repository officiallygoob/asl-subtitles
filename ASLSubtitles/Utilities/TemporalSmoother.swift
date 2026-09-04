import Foundation

/// Majority-vote temporal smoother to keep subtitles from flickering.
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

    /// Push a raw recognition; returns the display label + confidence to show.
    @discardableResult
    func push(_ result: RecognitionResult) -> (text: String, confidence: Double, isEmpty: Bool) {
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

        if let best, best.value >= minVotes {
            let confs = meaningful.filter { $0.label == best.key }.map(\.confidence)
            let avg = confs.reduce(0, +) / Double(max(confs.count, 1))
            currentLabel = displayForm(best.key)
            currentConfidence = avg
            lastAcceptedAt = Date()
        } else if Date().timeIntervalSince(lastAcceptedAt) > holdDuration {
            // Fade toward empty watching state.
            currentConfidence = max(0, currentConfidence - 0.08)
            if currentConfidence < 0.25 {
                currentLabel = ""
                currentConfidence = 0
            }
        }

        let watching = currentLabel.isEmpty
        return (currentLabel, currentConfidence, watching)
    }

    private func displayForm(_ raw: String) -> String {
        if raw.count == 1, raw.rangeOfCharacter(from: .letters) != nil {
            return raw.uppercased()
        }
        return raw.capitalized
    }
}
