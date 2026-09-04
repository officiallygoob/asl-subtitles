import Foundation

/// Records labeled landmark sequences for Create ML Hand Action / later fine-tunes.
///
/// Highest-leverage on-device next step toward friend-adapted recognition:
/// capture a few dozen examples per gloss, export JSON (and optional Create ML
/// friendly CSV), then train a small action classifier and drop it in as
/// `ASLSignClassifier.mlmodel`.
@MainActor
final class LandmarkRecorder: ObservableObject {
    @Published private(set) var isRecording = false
    @Published private(set) var frameCount = 0
    @Published var currentLabel: String = ""
    @Published private(set) var savedClipCount: Int = 0
    @Published private(set) var lastExportURL: URL?

    private var frames: [LandmarkFrame] = []
    private let maxFrames = 180  // ~6–12 s

    private var directory: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let dir = docs.appendingPathComponent("LandmarkRecordings", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func start(label: String) {
        let trimmed = label.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        currentLabel = trimmed
        frames.removeAll(keepingCapacity: true)
        frameCount = 0
        isRecording = true
    }

    func append(_ frame: LandmarkFrame) {
        guard isRecording else { return }
        frames.append(frame)
        if frames.count > maxFrames {
            frames.removeFirst(frames.count - maxFrames)
        }
        frameCount = frames.count
    }

    /// Stop and write JSON clip `{ label, frames: [...] }`.
    @discardableResult
    func stopAndSave() -> URL? {
        guard isRecording else { return nil }
        isRecording = false
        defer {
            frames.removeAll(keepingCapacity: true)
            frameCount = 0
        }
        guard frames.count >= 8 else { return nil }

        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let safeLabel = currentLabel.replacingOccurrences(of: "/", with: "-")
        let url = directory.appendingPathComponent("\(safeLabel)_\(stamp).json")

        let payload: [String: Any] = [
            "label": currentLabel,
            "exportedAt": ISO8601DateFormatter().string(from: Date()),
            "frameCount": frames.count,
            "schema": "asl-subtitles.landmark-sequence.v1",
            "frames": frames.map { frame -> [String: Any] in
                [
                    "timestamp": frame.timestamp,
                    "activity": frame.activity,
                    "hands": frame.hands.map { hand -> [String: Any] in
                        [
                            "chirality": hand.chirality,
                            "confidence": hand.confidence,
                            "joints": hand.joints
                        ]
                    },
                    "body": frame.body.map { ["name": $0.name, "x": $0.x, "y": $0.y, "confidence": $0.confidence] },
                    "face": frame.face.map { ["name": $0.name, "x": $0.x, "y": $0.y, "confidence": $0.confidence] }
                ]
            }
        ]

        do {
            let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: url, options: .atomic)
            savedClipCount += 1
            lastExportURL = url
            return url
        } catch {
            return nil
        }
    }

    func cancel() {
        isRecording = false
        frames.removeAll()
        frameCount = 0
    }

    /// List saved clip URLs (for sharing / Files app).
    func listClips() -> [URL] {
        (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil))?
            .filter { $0.pathExtension == "json" }
            .sorted(by: { $0.lastPathComponent > $1.lastPathComponent }) ?? []
    }

    /// Bundle all clips into one JSONL file for training pipelines.
    func exportJSONL() -> URL? {
        let clips = listClips()
        guard !clips.isEmpty else { return nil }
        let out = directory.appendingPathComponent("all_clips.jsonl")
        var lines: [String] = []
        for url in clips {
            if let s = try? String(contentsOf: url, encoding: .utf8) {
                // One clip per line (compact).
                if let data = s.data(using: .utf8),
                   let obj = try? JSONSerialization.jsonObject(with: data),
                   let compact = try? JSONSerialization.data(withJSONObject: obj),
                   let line = String(data: compact, encoding: .utf8) {
                    lines.append(line)
                }
            }
        }
        do {
            try lines.joined(separator: "\n").write(to: out, atomically: true, encoding: .utf8)
            lastExportURL = out
            return out
        } catch {
            return nil
        }
    }
}
