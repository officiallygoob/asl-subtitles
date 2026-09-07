import CoreML
import Foundation

/// On-device PoseLSTM / Create ML plug-in — **primary** recognition path.
///
/// Loads `ASLSignClassifier.mlpackage` / `.mlmodel` / `.mlmodelc` from the app
/// bundle or Documents. Expects input `poses` shaped `[1, T, 170]` (FEATURE_DIM v2:
/// hands + body + face + activity + NMM). Outputs logits or Create ML class probs.
///
/// Privacy: inference is entirely on-device. No network.
final class CoreMLSignClassifier {
    private var model: MLModel?
    private(set) var isAvailable = false
    private(set) var modelName: String = ""
    private var labels: [String] = []
    private let windowSize = 32
    private let featureDim = LandmarkFrame.featureDim

    /// Minimum confidence to prefer ML over heuristics.
    var confidenceThreshold: Double = 0.42


    init() {
        isAvailable = false
        loadIfPresent()
    }

    func loadIfPresent() {
        let candidates: [URL] = {
            var urls: [URL] = []
            let exts = ["mlpackage", "mlmodelc", "mlmodel"]
            for ext in exts {
                if let builtIn = Bundle.main.url(forResource: "ASLSignClassifier", withExtension: ext) {
                    urls.append(builtIn)
                }
            }
            // Nested Models/ group
            if let res = Bundle.main.resourceURL {
                for ext in exts {
                    urls.append(res.appendingPathComponent("ASLSignClassifier.\(ext)"))
                    urls.append(res.appendingPathComponent("Models/ASLSignClassifier.\(ext)"))
                }
            }
            if let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
                for ext in exts {
                    urls.append(docs.appendingPathComponent("ASLSignClassifier.\(ext)"))
                }
            }
            return urls
        }()

        for url in candidates where FileManager.default.fileExists(atPath: url.path) {
            do {
                let compiled: URL
                switch url.pathExtension {
                case "mlmodel":
                    compiled = try MLModel.compileModel(at: url)
                case "mlpackage":
                    // Xcode usually compiles packages; try direct load then compile.
                    if let direct = try? MLModel(contentsOf: url) {
                        model = direct
                        isAvailable = true
                        modelName = url.lastPathComponent
                        loadLabels(beside: url)
                        return
                    }
                    compiled = try MLModel.compileModel(at: url)
                default:
                    compiled = url
                }
                let ml = try MLModel(contentsOf: compiled)
                model = ml
                isAvailable = true
                modelName = url.lastPathComponent
                loadLabels(beside: url)
                return
            } catch {
                continue
            }
        }
        model = nil
        isAvailable = false
        modelName = ""
        labels = []
    }

    private func loadLabels(beside url: URL) {
        let side = url.deletingLastPathComponent().appendingPathComponent("ASLSignClassifier.labels.json")
        let alt = url.deletingPathExtension().appendingPathExtension("labels.json")
        for candidate in [side, alt] {
            if let data = try? Data(contentsOf: candidate),
               let parsed = Self.parseLabelList(from: data) {
                labels = parsed
                return
            }
        }
        // Metadata embedded by coremltools
        if let meta = model?.modelDescription.metadata[.creatorDefinedKey] as? [String: String],
           let raw = meta["labels"],
           let data = raw.data(using: .utf8),
           let labs = Self.parseLabelList(from: data) {
            labels = labs
        }
    }

    /// Accepts `["HELLO", …]` or `{"labels":[…]}` (training script / older exports).
    private static func parseLabelList(from data: Data) -> [String]? {
        guard let obj = try? JSONSerialization.jsonObject(with: data) else { return nil }
        if let labs = obj as? [String], !labs.isEmpty { return labs }
        if let dict = obj as? [String: Any], let labs = dict["labels"] as? [String], !labs.isEmpty {
            return labs
        }
        return nil
    }

    /// Classify a temporal window of landmark feature vectors (preferably FEATURE_DIM=170).
    /// Previous accepted gloss for on-device bigram top-k rerank (nil = plain argmax).
    var previousGloss: String?

    /// Val-gated multi-crop count (WLASL100 val selected nc=5 @ max context 96).
    private let multiCropCount = 5
    /// Match utterance buffer cap used in ASLSessionController.
    private let maxContextFrames = 96

    func classify(window: [[Double]]) -> RecognitionResult? {
        guard isAvailable, let model, !window.isEmpty else { return nil }

        let normalized = normalizeDims(window)
        let crops = temporalCrops(normalized, cropCount: multiCropCount, maxContext: maxContextFrames)
        guard !crops.isEmpty else { return nil }

        do {
            let inputName = preferredInputName(model)
            var acc: [Double]?
            var nOk = 0
            for crop in crops {
                let prepared = padOrTrim32(crop)
                let t = prepared.count
                let d = prepared[0].count
                let flat = prepared.flatMap { $0 }
                guard d > 0, flat.count == t * d else { continue }
                let arr = try MLMultiArray(shape: [1, NSNumber(value: t), NSNumber(value: d)], dataType: .float32)
                for i in 0..<flat.count {
                    arr[i] = NSNumber(value: Float(flat[i]))
                }
                let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: arr)])
                let out = try model.prediction(from: provider)
                guard let logits = extractLogits(out) else {
                    // Fall back to single-crop dictionary parse if logits unavailable.
                    if crops.count == 1 { return parseOutput(out) }
                    continue
                }
                if acc == nil {
                    acc = logits
                } else if acc!.count == logits.count {
                    for i in 0..<logits.count { acc![i] += logits[i] }
                } else {
                    continue
                }
                nOk += 1
            }
            guard var summed = acc, !summed.isEmpty, nOk > 0 else { return nil }
            let inv = 1.0 / Double(nOk)
            for i in 0..<summed.count { summed[i] *= inv }
            return resultFromLogitsArray(summed)
        } catch {
            return nil
        }
    }

    /// Evenly spaced 32-frame crops over up to `maxContext` trailing frames (val-gated).
    private func temporalCrops(_ frames: [[Double]], cropCount: Int, maxContext: Int) -> [[[Double]]] {
        var seq = frames
        if seq.count > maxContext {
            seq = Array(seq.suffix(maxContext))
        }
        if seq.count <= windowSize {
            return [seq]
        }
        let maxStart = seq.count - windowSize
        let n = max(1, cropCount)
        if n == 1 {
            return [Array(seq.suffix(windowSize))]
        }
        var starts = Set<Int>()
        for i in 0..<n {
            let s = Int((Double(i) * Double(maxStart) / Double(n - 1)).rounded())
            starts.insert(min(max(s, 0), maxStart))
        }
        return starts.sorted().map { Array(seq[$0 ..< ($0 + windowSize)]) }
    }

    private func normalizeDims(_ window: [[Double]]) -> [[Double]] {
        var frames = window
        if let d = frames.first?.count, d != featureDim, d < 100 {
            frames = frames.map { row in
                if row.count >= featureDim { return Array(row.prefix(featureDim)) }
                return row + Array(repeating: 0.0, count: featureDim - row.count)
            }
        } else if let d = frames.first?.count, d > featureDim {
            frames = frames.map { Array($0.prefix(featureDim)) }
        } else if let d = frames.first?.count, d < featureDim {
            frames = frames.map { $0 + Array(repeating: 0.0, count: featureDim - $0.count) }
        }
        return frames
    }

    private func padOrTrim32(_ window: [[Double]]) -> [[Double]] {
        var frames = window
        if frames.count < windowSize {
            let pad = Array(repeating: frames.first ?? Array(repeating: 0.0, count: featureDim), count: windowSize - frames.count)
            frames = pad + frames
        } else if frames.count > windowSize {
            frames = Array(frames.suffix(windowSize))
        }
        return frames
    }

    private func extractLogits(_ out: MLFeatureProvider) -> [Double]? {
        for name in ["logits", "output", "Identity", "var_40"] {
            if let arr = out.featureValue(for: name)?.multiArrayValue, arr.count > 1 {
                return (0..<arr.count).map { arr[$0].doubleValue }
            }
        }
        for name in out.featureNames {
            if let arr = out.featureValue(for: name)?.multiArrayValue, arr.count > 1 {
                // Skip tiny vectors / scalar probs
                if arr.count >= 8 { return (0..<arr.count).map { arr[$0].doubleValue } }
            }
        }
        return nil
    }

    private func resultFromLogitsArray(_ logits: [Double]) -> RecognitionResult? {
        let n = logits.count
        guard n > 0 else { return nil }
        let maxL = logits.max() ?? 0
        var exps = logits.map { exp($0 - maxL) }
        let sum = exps.reduce(0, +)
        guard sum > 0 else { return nil }
        exps = exps.map { $0 / sum }
        let bestIdx: Int
        if labels.count == exps.count {
            bestIdx = GlossBigramPrior.rerank(probs: exps, labels: labels, prevGloss: previousGloss)
        } else if let idx = exps.indices.max(by: { exps[$0] < exps[$1] }) {
            bestIdx = idx
        } else {
            return nil
        }
        let conf = exps[bestIdx]
        guard conf >= confidenceThreshold else { return nil }
        let label: String
        if bestIdx < labels.count {
            label = labels[bestIdx]
        } else {
            label = "CLASS_\(bestIdx)"
        }
        return RecognitionResult(
            label: label.uppercased(),
            kind: .everydaySign,
            confidence: conf,
            timestamp: Date(),
            gloss: label.uppercased()
        )
    }

    private func preferredInputName(_ model: MLModel) -> String {
        let names = model.modelDescription.inputDescriptionsByName.keys
        for candidate in ["poses", "multiArrayInput", "input", "landmarks", "sequence"] {
            if names.contains(candidate) { return candidate }
        }
        return names.first ?? "input"
    }

    private func parseOutput(_ out: MLFeatureProvider) -> RecognitionResult? {
        // Create ML style dictionary
        for name in out.featureNames {
            if let value = out.featureValue(for: name),
               let dict = value.dictionaryValue as? [AnyHashable: NSNumber],
               !dict.isEmpty {
                let best = dict.max(by: { $0.value.doubleValue < $1.value.doubleValue })
                if let best {
                    let label = String(describing: best.key)
                    let conf = best.value.doubleValue
                    guard conf >= confidenceThreshold, !label.isEmpty else { return nil }
                    return RecognitionResult(
                        label: label.uppercased(),
                        kind: .everydaySign,
                        confidence: conf,
                        timestamp: Date(),
                        gloss: label.uppercased()
                    )
                }
            }
        }

        // Logits multiarray → softmax
        for name in ["logits", "output", "Identity", "var_40"] {
            if let arr = out.featureValue(for: name)?.multiArrayValue {
                return resultFromLogits(arr)
            }
        }
        for name in out.featureNames {
            if let arr = out.featureValue(for: name)?.multiArrayValue, arr.count > 1 {
                if let r = resultFromLogits(arr) { return r }
            }
        }

        for name in ["label", "classLabel", "target"] {
            if let value = out.featureValue(for: name)?.stringValue, !value.isEmpty {
                return RecognitionResult(
                    label: value.uppercased(),
                    kind: .everydaySign,
                    confidence: 0.7,
                    timestamp: Date(),
                    gloss: value.uppercased()
                )
            }
        }
        return nil
    }

    private func resultFromLogits(_ arr: MLMultiArray) -> RecognitionResult? {
        let logits = (0..<arr.count).map { arr[$0].doubleValue }
        return resultFromLogitsArray(logits)
    }
}
