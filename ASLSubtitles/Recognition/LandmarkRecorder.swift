import Foundation

/// On-device Train / Capture for optional friend adaptation.
///
/// Primary accuracy ships via bundled Core ML trained offline on public pose
/// dumps. This recorder lets you capture labeled landmark sequences
/// (hands+face+body+NMM) for Create ML Hand Action or PoseLSTM fine-tune.
/// Clips never leave the device unless you export/share them.
@MainActor
final class LandmarkRecorder: ObservableObject {
    @Published private(set) var isRecording = false
    @Published private(set) var frameCount = 0
    @Published var currentLabel: String = ""
    @Published private(set) var savedClipCount: Int = 0
    @Published private(set) var takesForCurrentLabel: Int = 0
    @Published private(set) var lastExportURL: URL?
    @Published var targetTakes: Int = 5

    private var frames: [LandmarkFrame] = []
    private let maxFrames = 180

    private var directory: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let dir = docs.appendingPathComponent("LandmarkRecordings", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func start(label: String) {
        let trimmed = label.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        if trimmed.uppercased() != currentLabel.uppercased() {
            currentLabel = trimmed
            takesForCurrentLabel = countTakes(for: trimmed)
        } else {
            currentLabel = trimmed
        }
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
        let take = takesForCurrentLabel + 1
        let url = directory.appendingPathComponent("\(safeLabel)_take\(take)_\(stamp).json")

        let payload: [String: Any] = [
            "label": currentLabel.uppercased(),
            "take": take,
            "exportedAt": ISO8601DateFormatter().string(from: Date()),
            "frameCount": frames.count,
            "schema": "asl-subtitles.landmark-sequence.v2",
            "featureLayoutVersion": LandmarkFrame.featureLayoutVersion,
            "featureDim": LandmarkFrame.featureDim,
            "exportFormats": [
                "poseLSTM": "server/scripts/finetune_from_recordings.py",
                "createMLHandAction": "exportCreateMLCSV()"
            ],
            "frames": frames.map { frame -> [String: Any] in
                [
                    "timestamp": frame.timestamp,
                    "activity": frame.activity,
                    "nmm": frame.nmm ?? Array(repeating: 0, count: LandmarkFrame.nmmChannelOrder.count),
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
            takesForCurrentLabel = take
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

    func listClips() -> [URL] {
        (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil))?
            .filter { $0.pathExtension == "json" }
            .sorted(by: { $0.lastPathComponent > $1.lastPathComponent }) ?? []
    }

    func countTakes(for label: String) -> Int {
        let prefix = label.uppercased().replacingOccurrences(of: "/", with: "-")
        return listClips().filter { $0.lastPathComponent.uppercased().hasPrefix(prefix) }.count
    }

    func exportJSONL() -> URL? {
        let clips = listClips()
        guard !clips.isEmpty else { return nil }
        let out = directory.appendingPathComponent("all_clips.jsonl")
        var lines: [String] = []
        for url in clips {
            if let s = try? String(contentsOf: url, encoding: .utf8),
               let data = s.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: data),
               let compact = try? JSONSerialization.data(withJSONObject: obj),
               let line = String(data: compact, encoding: .utf8) {
                lines.append(line)
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

    func exportCreateMLCSV() -> URL? {
        let clips = listClips()
        guard !clips.isEmpty else { return nil }
        let out = directory.appendingPathComponent("create_ml_hand_action.csv")
        var rows: [String] = ["Action,Time,wrist_x,wrist_y,thumbTip_x,thumbTip_y,indexTip_x,indexTip_y,middleTip_x,middleTip_y,ringTip_x,ringTip_y,littleTip_x,littleTip_y"]
        for url in clips {
            guard let data = try? Data(contentsOf: url),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let label = obj["label"] as? String,
                  let frames = obj["frames"] as? [[String: Any]] else { continue }
            for frame in frames {
                let t = frame["timestamp"] as? Double ?? 0
                let hands = frame["hands"] as? [[String: Any]] ?? []
                let right = hands.first(where: { ($0["chirality"] as? String) == "right" }) ?? hands.first
                let joints = right?["joints"] as? [String: [Double]] ?? [:]
                func xy(_ name: String) -> String {
                    let v = joints[name] ?? [0, 0]
                    let x = v.count > 0 ? v[0] : 0
                    let y = v.count > 1 ? v[1] : 0
                    return "\(x),\(y)"
                }
                rows.append("\(label),\(t),\(xy("wrist")),\(xy("thumbTip")),\(xy("indexTip")),\(xy("middleTip")),\(xy("ringTip")),\(xy("littleTip"))")
            }
        }
        do {
            try rows.joined(separator: "\n").write(to: out, atomically: true, encoding: .utf8)
            lastExportURL = out
            return out
        } catch {
            return nil
        }
    }
}
