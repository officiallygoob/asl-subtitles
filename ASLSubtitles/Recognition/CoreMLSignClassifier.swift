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
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let labs = obj["labels"] as? [String] {
                labels = labs
                return
            }
        }
        // Metadata embedded by coremltools
        if let meta = model?.modelDescription.metadata[.creatorDefinedKey] as? [String: String],
           let raw = meta["labels"],
           let data = raw.data(using: .utf8),
           let labs = try? JSONSerialization.jsonObject(with: data) as? [String] {
            labels = labs
        }
    }

    /// Classify a temporal window of landmark feature vectors (preferably FEATURE_DIM=170).
    func classify(window: [[Double]]) -> RecognitionResult? {
        guard isAvailable, let model, !window.isEmpty else { return nil }

        // Resample / pad to fixed 32 × D
        let prepared = prepareWindow(window)
        let t = prepared.count
        let d = prepared[0].count
        let flat = prepared.flatMap { $0 }
        guard d > 0, flat.count == t * d else { return nil }

        do {
            let arr = try MLMultiArray(shape: [1, NSNumber(value: t), NSNumber(value: d)], dataType: .float32)
            for i in 0..<flat.count {
                arr[i] = NSNumber(value: Float(flat[i]))
            }

            let inputName = preferredInputName(model)
            let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: arr)])
            let out = try model.prediction(from: provider)
            return parseOutput(out)
        } catch {
            return nil
        }
    }

    private func prepareWindow(_ window: [[Double]]) -> [[Double]] {
        var frames = window
        // If hand-only (~47 dims), refuse — caller should pass holistic vectors.
        if let d = frames.first?.count, d != featureDim, d < 100 {
            // Still try: zero-pad to featureDim so an old Create ML hand model can fail softly.
            frames = frames.map { row in
                if row.count >= featureDim { return Array(row.prefix(featureDim)) }
                return row + Array(repeating: 0.0, count: featureDim - row.count)
            }
        } else if let d = frames.first?.count, d > featureDim {
            frames = frames.map { Array($0.prefix(featureDim)) }
        } else if let d = frames.first?.count, d < featureDim {
            frames = frames.map { $0 + Array(repeating: 0.0, count: featureDim - $0.count) }
        }
        if frames.count < windowSize {
            let pad = Array(repeating: frames.first ?? Array(repeating: 0.0, count: featureDim), count: windowSize - frames.count)
            frames = pad + frames
        } else if frames.count > windowSize {
            frames = Array(frames.suffix(windowSize))
        }
        return frames
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
        let n = arr.count
        guard n > 0 else { return nil }
        var logits = [Double](repeating: 0, count: n)
        for i in 0..<n { logits[i] = arr[i].doubleValue }
        let maxL = logits.max() ?? 0
        var exps = logits.map { exp($0 - maxL) }
        let sum = exps.reduce(0, +)
        guard sum > 0 else { return nil }
        exps = exps.map { $0 / sum }
        guard let bestIdx = exps.indices.max(by: { exps[$0] < exps[$1] }) else { return nil }
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
}
