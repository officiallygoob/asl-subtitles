import Foundation

/// Local spaced-ish progress for Learn ASL cards (on-device only).
struct LearnCardProgress: Codable, Hashable {
    var gloss: String
    var correctCount: Int = 0
    var wrongCount: Int = 0
    var streak: Int = 0
    /// Lower = due sooner.
    var ease: Double = 2.2
    var dueAt: Date = Date()
    var lastResult: Bool? = nil
}

@MainActor
final class LearnProgressStore: ObservableObject {
    static let shared = LearnProgressStore()

    @Published private(set) var progress: [String: LearnCardProgress] = [:]

    private var fileURL: URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        let folder = dir.appendingPathComponent("ASLSubtitles", isDirectory: true)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder.appendingPathComponent("learn_progress.json")
    }

    init() { load() }

    func load() {
        guard let data = try? Data(contentsOf: fileURL),
              let decoded = try? JSONDecoder().decode([String: LearnCardProgress].self, from: data) else {
            progress = [:]
            return
        }
        progress = decoded
    }

    func record(gloss: String, correct: Bool) {
        var item = progress[gloss] ?? LearnCardProgress(gloss: gloss)
        if correct {
            item.correctCount += 1
            item.streak += 1
            item.ease = min(3.0, item.ease + 0.15)
            let hours = pow(item.ease, Double(min(item.streak, 5))) * 2
            item.dueAt = Date().addingTimeInterval(hours * 3600)
        } else {
            item.wrongCount += 1
            item.streak = 0
            item.ease = max(1.3, item.ease - 0.2)
            item.dueAt = Date().addingTimeInterval(10 * 60) // retry in 10 min
        }
        item.lastResult = correct
        progress[gloss] = item
        persist()
    }

    func clear() {
        progress = [:]
        persist()
    }

    var masteredCount: Int {
        progress.values.filter { $0.correctCount >= 2 && $0.streak >= 1 }.count
    }

    private func persist() {
        guard let data = try? JSONEncoder().encode(progress) else { return }
        try? data.write(to: fileURL, options: [.atomic])
    }
}
