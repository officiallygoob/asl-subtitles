import UIKit
import Foundation
import SwiftUI
import Vision

/// Flashcard learn loop: show card → ready → watch signs → score via SignRecognizer.
@MainActor
final class LearnSessionController: ObservableObject {
    enum Phase: Equatable {
        case prompt
        case watching
        case correct
        case tryAgain
    }

    @Published var phase: Phase = .prompt
    @Published var queue: [LearnCard] = []
    @Published var index: Int = 0
    @Published var feedback: String = ""
    @Published var liveGuess: String = ""
    @Published var liveConfidence: Double = 0
    @Published var holdProgress: Double = 0

    private let recognizer = SignRecognizer()
    private let smoother = TemporalSmoother(windowSize: 10, minVotes: 5, holdDuration: 0.8)
    private let progressStore = LearnProgressStore.shared
    private var stableTicks = 0
    private let neededStableTicks = 6
    private let confidenceThreshold = 0.62

    var current: LearnCard? {
        guard !queue.isEmpty, index >= 0, index < queue.count else { return nil }
        return queue[index]
    }

    var progressLabel: String {
        guard !queue.isEmpty else { return "0 / 0" }
        return "\(min(index + 1, queue.count)) / \(queue.count)"
    }

    func startDeck() {
        recognizer.reset()
        smoother.reset()
        queue = LearnDeck.orderedQueue(progress: progressStore.progress)
        index = 0
        phase = .prompt
        feedback = ""
        liveGuess = ""
        holdProgress = 0
        stableTicks = 0
    }

    func imReady() {
        guard current != nil else { return }
        recognizer.reset()
        smoother.reset()
        phase = .watching
        feedback = "Sign it now…"
        liveGuess = ""
        holdProgress = 0
        stableTicks = 0
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
    }

    func skip() {
        advance(mark: nil)
    }

    func gotItManual() {
        guard let card = current else { return }
        progressStore.record(gloss: card.gloss, correct: true)
        phase = .correct
        feedback = "Marked as learned"
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.55) { [weak self] in
            self?.advance(mark: nil)
        }
    }

    func again() {
        guard let card = current else { return }
        progressStore.record(gloss: card.gloss, correct: false)
        // Put card near front of remaining queue
        if index + 1 < queue.count {
            let card = queue.remove(at: index)
            let insertAt = min(index + 2, queue.count)
            queue.insert(card, at: insertAt)
        }
        phase = .prompt
        feedback = "We’ll try that one again soon."
        holdProgress = 0
    }

    /// Feed a holistic frame while watching.
    func consume(frame: LandmarkFrame) {
        guard phase == .watching, let card = current else { return }
        let result = recognizer.recognize(frame: frame)
        let smoothed = smoother.push(result)
        liveGuess = smoothed.text
        liveConfidence = smoothed.confidence

        let matched = matches(card: card, result: result, display: smoothed.text)
        if matched && smoothed.confidence >= confidenceThreshold {
            stableTicks += 1
            holdProgress = Double(stableTicks) / Double(neededStableTicks)
            if stableTicks >= neededStableTicks {
                progressStore.record(gloss: card.gloss, correct: true)
                phase = .correct
                feedback = encouragingCorrect(for: card)
                UINotificationFeedbackGenerator().notificationOccurred(.success)
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.9) { [weak self] in
                    self?.advance(mark: nil)
                }
            }
        } else {
            stableTicks = max(0, stableTicks - 1)
            holdProgress = Double(stableTicks) / Double(neededStableTicks)
            if !smoothed.isEmpty && smoothed.confidence >= 0.55 && !matched {
                // Soft “try again” hint without failing the card immediately
                feedback = "Looking like \(smoothed.text) — keep going for \(card.title)"
            }
        }
    }

    private func matches(card: LearnCard, result: RecognitionResult, display: String) -> Bool {
        let target = card.gloss.uppercased()
        let gloss = result.gloss.uppercased()
        let label = result.label.uppercased()
        let disp = display.uppercased()
        if card.isLetter {
            let letter = target
            return gloss == letter || label == letter || disp == letter
                || gloss.hasPrefix(letter) || disp.hasPrefix(letter)
        }
        // Compare gloss tokens / English phrase contains
        if gloss == target || gloss.replacingOccurrences(of: " ", with: "-") == target { return true }
        if label.replacingOccurrences(of: " ", with: "-").uppercased() == target { return true }
        let targetWords = target.replacingOccurrences(of: "-", with: " ")
        if disp.replacingOccurrences(of: "-", with: " ").contains(targetWords) { return true }
        // Phrase map: HELLO vs Hello.
        if disp.replacingOccurrences(of: ".", with: "").trimmingCharacters(in: .whitespaces)
            .caseInsensitiveCompare(card.title) == .orderedSame { return true }
        return false
    }

    private func encouragingCorrect(for card: LearnCard) -> String {
        let lines = ["Nice!", "You got it!", "Clear signing!", "Great!", "Yes!"]
        return lines.randomElement()! + " \(card.title)"
    }

    private func advance(mark: Bool?) {
        recognizer.reset()
        smoother.reset()
        stableTicks = 0
        holdProgress = 0
        liveGuess = ""
        if index + 1 < queue.count {
            index += 1
            phase = .prompt
            feedback = ""
        } else {
            // Reshuffle due cards
            queue = LearnDeck.orderedQueue(progress: progressStore.progress)
            index = 0
            phase = .prompt
            feedback = "Deck refreshed — keep practicing!"
        }
    }
}
