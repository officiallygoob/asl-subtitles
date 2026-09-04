import CoreML
import Foundation

/// Optional Core ML plug-in behind `SignRecognizer`.
///
/// Drop a Create ML Hand Action / custom landmark-sequence `.mlmodelc` into the
/// app bundle (or Documents) named `ASLSignClassifier.mlmodelc` and this class
/// will score windows when confident. Heuristics remain the offline fallback.
///
/// We do **not** ship fake Core ML binaries — if no model is present, `classify`
/// returns nil and the caller falls through.
final class CoreMLSignClassifier {
    private var model: MLModel?
    private(set) var isAvailable = false
    private(set) var modelName: String = ""

    /// Minimum confidence to prefer ML over heuristics.
    var confidenceThreshold: Double = 0.65

    init() {
        loadIfPresent()
    }

    func loadIfPresent() {
        let candidates: [URL] = {
            var urls: [URL] = []
            if let builtIn = Bundle.main.url(forResource: "ASLSignClassifier", withExtension: "mlmodelc") {
                urls.append(builtIn)
            }
            if let builtIn2 = Bundle.main.url(forResource: "ASLSignClassifier", withExtension: "mlmodel") {
                urls.append(builtIn2)
            }
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            if let docs {
                urls.append(docs.appendingPathComponent("ASLSignClassifier.mlmodelc"))
                urls.append(docs.appendingPathComponent("ASLSignClassifier.mlmodel"))
            }
            return urls
        }()

        for url in candidates where FileManager.default.fileExists(atPath: url.path) {
            do {
                let compiled: URL
                if url.pathExtension == "mlmodel" {
                    compiled = try MLModel.compileModel(at: url)
                } else {
                    compiled = url
                }
                let ml = try MLModel(contentsOf: compiled)
                model = ml
                isAvailable = true
                modelName = url.lastPathComponent
                return
            } catch {
                continue
            }
        }
        model = nil
        isAvailable = false
        modelName = ""
    }

    /// Classify a temporal window of landmark feature vectors.
    /// Expected input names are flexible: tries `poses`, `multiArray`, `input`.
    func classify(window: [[Double]]) -> RecognitionResult? {
        guard isAvailable, let model, !window.isEmpty else { return nil }

        let flat = window.flatMap { $0 }
        let t = window.count
        let d = window[0].count
        guard d > 0, flat.count == t * d else { return nil }

        do {
            let arr = try MLMultiArray(shape: [1, NSNumber(value: t), NSNumber(value: d)], dataType: .double)
            for i in 0..<flat.count {
                arr[i] = NSNumber(value: flat[i])
            }

            let inputName = preferredInputName(model)
            let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: arr)])
            let out = try model.prediction(from: provider)
            return parseOutput(out)
        } catch {
            // Shape mismatch is common until the user exports a matching Create ML model.
            return nil
        }
    }

    private func preferredInputName(_ model: MLModel) -> String {
        let names = model.modelDescription.inputDescriptionsByName.keys
        for candidate in ["poses", "multiArrayInput", "input", "landmarks", "sequence"] {
            if names.contains(candidate) { return candidate }
        }
        return names.first ?? "input"
    }

    private func parseOutput(_ out: MLFeatureProvider) -> RecognitionResult? {
        // Prefer labeled class + confidence dictionary (Create ML style).
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
                        label: label,
                        kind: .everydaySign,
                        confidence: conf,
                        timestamp: Date()
                    )
                }
            }
        }
        // Or a single string label feature.
        for name in ["label", "classLabel", "target"] {
            if let value = out.featureValue(for: name)?.stringValue, !value.isEmpty {
                return RecognitionResult(
                    label: value,
                    kind: .everydaySign,
                    confidence: 0.7,
                    timestamp: Date()
                )
            }
        }
        return nil
    }
}
