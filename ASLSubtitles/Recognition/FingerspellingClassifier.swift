import Foundation
import Vision

/// Rule-based ASL fingerspelling A–Z from a static hand shape.
///
/// Honest MVP: many letters rely on subtle motion or side views that a single
/// front-facing frame cannot resolve perfectly. Confidence reflects that.
///
/// TODO(CoreML): Swap `classify` body for an MLModel prediction on
/// `LandmarkFeatures.featureVector()` while keeping this API stable.
struct FingerspellingClassifier {

    func classify(_ features: LandmarkFeatures) -> RecognitionResult? {
        let candidates = letterScores(features)
        guard let best = candidates.max(by: { $0.score < $1.score }),
              best.score >= 0.5 else {
            return nil
        }
        return RecognitionResult(
            label: best.letter,
            kind: .fingerspell,
            confidence: best.score,
            timestamp: Date()
        )
    }

    private struct Candidate {
        let letter: String
        let score: Double
    }

    private func letterScores(_ f: LandmarkFeatures) -> [Candidate] {
        var out: [Candidate] = []

        // A — fist with thumb alongside
        if f.isFist && f.thumbExtended && !f.indexExtended {
            out.append(.init(letter: "A", score: 0.72))
        }

        // B — flat hand, four fingers up, thumb tucked
        if f.nonThumbExtendedCount == 4 && !f.thumbExtended {
            out.append(.init(letter: "B", score: 0.78))
        }

        // C — curved open hand (pinch not closed, fingers partly bent)
        if f.nonThumbExtendedCount >= 2 && f.nonThumbExtendedCount <= 4,
           !f.thumbIndexPinch,
           let thumb = f.tip(.thumbTip), let index = f.tip(.indexTip),
           f.distance(thumb, index) > 0.05 && f.distance(thumb, index) < 0.18 {
            out.append(.init(letter: "C", score: 0.62))
        }

        // D — index up, others closed, thumb touches middle
        if f.indexExtended && !f.middleExtended && !f.ringExtended && !f.littleExtended {
            if f.thumbIndexPinch {
                out.append(.init(letter: "D", score: 0.55))
            } else {
                out.append(.init(letter: "D", score: 0.68))
                out.append(.init(letter: "Z", score: 0.40)) // static Z looks like D
            }
        }

        // E — curled fingertips, thumb across
        if f.tipsNearPalm && !f.indexExtended && !f.middleExtended {
            out.append(.init(letter: "E", score: 0.58))
        }

        // F — OK / thumb-index circle, other three up
        if f.thumbIndexPinch && f.middleExtended && f.ringExtended && f.littleExtended {
            out.append(.init(letter: "F", score: 0.8))
        }

        // G — index + thumb pointing sideways (approx: only index+thumb)
        if f.indexExtended && f.thumbExtended && !f.middleExtended && !f.ringExtended && !f.littleExtended {
            out.append(.init(letter: "G", score: 0.6))
            out.append(.init(letter: "Q", score: 0.45))
        }

        // H — index + middle together extended
        if f.indexExtended && f.middleExtended && !f.ringExtended && !f.littleExtended {
            out.append(.init(letter: "H", score: 0.7))
            out.append(.init(letter: "U", score: 0.55))
            out.append(.init(letter: "V", score: 0.5))
            out.append(.init(letter: "R", score: 0.42))
        }

        // I — pinky only
        if f.littleExtended && !f.indexExtended && !f.middleExtended && !f.ringExtended {
            out.append(.init(letter: "I", score: 0.78))
            out.append(.init(letter: "J", score: 0.4)) // J needs motion
        }

        // K — index+middle up, thumb between (hard); approximate as V with thumb out
        if f.indexExtended && f.middleExtended && f.thumbExtended && !f.ringExtended && !f.littleExtended {
            out.append(.init(letter: "K", score: 0.55))
            out.append(.init(letter: "V", score: 0.6))
        }

        // L — L shape: index + thumb
        if f.indexExtended && f.thumbExtended && !f.middleExtended && !f.ringExtended && !f.littleExtended {
            // Prefer L when angle at MCP is wide
            if let tip = f.tip(.indexTip), let mcp = f.hand.point(.indexMCP), let thumb = f.tip(.thumbTip) {
                let ang = f.angle(a: tip, b: mcp, c: thumb)
                out.append(.init(letter: "L", score: ang > 50 ? 0.82 : 0.65))
            } else {
                out.append(.init(letter: "L", score: 0.7))
            }
        }

        // M — three fingers over thumb (fist-like)
        if f.isFist && !f.thumbExtended {
            out.append(.init(letter: "M", score: 0.5))
            out.append(.init(letter: "N", score: 0.48))
            out.append(.init(letter: "S", score: 0.55))
            out.append(.init(letter: "T", score: 0.45))
            out.append(.init(letter: "E", score: 0.42))
        }

        // O — thumb-index pinch, others curled
        if f.thumbIndexPinch && !f.middleExtended && !f.ringExtended {
            out.append(.init(letter: "O", score: 0.75))
        }

        // P — similar to K pointing down (static ambiguity)
        // Q — similar to G

        // S — fist, thumb across front
        if f.isFist {
            out.append(.init(letter: "S", score: 0.52))
        }

        // T — fist, thumb between index/middle (approx fist)
        // already covered under M/N/S

        // U — index+middle together (see H)
        // V — index+middle spread
        if f.indexExtended && f.middleExtended && !f.ringExtended && !f.littleExtended,
           let i = f.tip(.indexTip), let m = f.tip(.middleTip) {
            let sep = f.distance(i, m)
            if sep > 0.06 {
                out.append(.init(letter: "V", score: 0.78))
            } else {
                out.append(.init(letter: "U", score: 0.72))
            }
        }

        // W — index+middle+ring
        if f.indexExtended && f.middleExtended && f.ringExtended && !f.littleExtended {
            out.append(.init(letter: "W", score: 0.8))
        }

        // X — hooked index (index partly bent)
        if !f.indexExtended && !f.middleExtended && !f.ringExtended && !f.littleExtended,
           let tip = f.tip(.indexTip), let pip = f.hand.point(.indexPIP), let mcp = f.hand.point(.indexMCP) {
            let bent = f.distance(tip, mcp) < f.distance(pip, mcp) * 1.4
            if bent {
                out.append(.init(letter: "X", score: 0.55))
            }
        }

        // Y — thumb + pinky
        if f.thumbExtended && f.littleExtended && !f.indexExtended && !f.middleExtended && !f.ringExtended {
            out.append(.init(letter: "Y", score: 0.85))
        }

        // Open palm can also look like "B" already handled

        return out
    }
}
