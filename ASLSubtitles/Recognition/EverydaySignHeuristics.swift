import Foundation
import Vision

/// Heuristic recognizer for a small everyday ASL vocabulary.
///
/// Signs that need motion use simple frame-to-frame deltas supplied by the caller.
/// This is intentionally limited — not fluent ASL.
///
/// TODO(CoreML): Move each rule into labeled training clips and train a sequence
/// model (or per-sign Create ML Hand Action classifier). Keep `classify` signature.
struct EverydaySignHeuristics {

    struct MotionContext {
        /// Change in hand center Y since last sample (Vision coords: up = +).
        var deltaY: CGFloat
        /// Change in hand center X.
        var deltaX: CGFloat
        /// Recent open-palm wave oscillation count estimate.
        var lateralOscillations: Int
    }

    func classify(hands: [LandmarkFeatures], motion: MotionContext) -> RecognitionResult? {
        guard let primary = hands.first else { return nil }

        var candidates: [(String, Double)] = []

        // HELLO — open palm near head height, small side motion / wave
        if primary.isOpenPalm && primary.handCenterY > 0.55 {
            let waveBoost = motion.lateralOscillations >= 1 || abs(motion.deltaX) > 0.03
            candidates.append(("HELLO", waveBoost ? 0.8 : 0.62))
        }

        // BYE — similar to hello / open palm wave
        if primary.isOpenPalm && abs(motion.deltaX) > 0.04 {
            candidates.append(("BYE", 0.7))
        }

        // THANKS — flat hand from chin/mouth area moving forward/down
        if primary.isOpenPalm || (primary.nonThumbExtendedCount >= 3) {
            if primary.handCenterY > 0.5 && motion.deltaY < -0.025 {
                candidates.append(("THANKS", 0.75))
            }
        }

        // YES — fist nodding (vertical bob)
        if primary.isFist && abs(motion.deltaY) > 0.03 {
            candidates.append(("YES", 0.72))
        }

        // NO — pinch / two-finger snap sideways (index+middle + thumb, lateral)
        if primary.thumbExtended && primary.indexExtended && primary.middleExtended
            && !primary.ringExtended && abs(motion.deltaX) > 0.025 {
            candidates.append(("NO", 0.68))
        }
        // Alternate NO: thumb-index pinch flick
        if primary.thumbIndexPinch && abs(motion.deltaX) > 0.03 {
            candidates.append(("NO", 0.6))
        }

        // PLEASE — open palm on chest circling (approx: open palm mid torso + motion)
        if primary.isOpenPalm && primary.handCenterY > 0.3 && primary.handCenterY < 0.65 {
            if hypot(motion.deltaX, motion.deltaY) > 0.02 {
                candidates.append(("PLEASE", 0.65))
            }
        }

        // HELP — fist on open palm (two hands) lifting
        if hands.count >= 2 {
            let a = hands[0]
            let b = hands[1]
            if (a.isFist && b.isOpenPalm) || (b.isFist && a.isOpenPalm) {
                let lift = motion.deltaY > 0.02
                candidates.append(("HELP", lift ? 0.78 : 0.6))
            }
        }

        // NAME — H / U handshape tapping (index+middle extended, short motion)
        if primary.indexExtended && primary.middleExtended && !primary.ringExtended {
            if hypot(motion.deltaX, motion.deltaY) > 0.015 {
                candidates.append(("NAME", 0.58))
            }
        }

        // FRIEND — interlocking index hooks (two hands, index extended)
        if hands.count >= 2 {
            let a = hands[0]
            let b = hands[1]
            if a.indexExtended && b.indexExtended && !a.middleExtended && !b.middleExtended {
                candidates.append(("FRIEND", 0.7))
            }
        }

        // LOVE — arms crossed / both fists at chest (approx: two fists mid)
        if hands.count >= 2 {
            let a = hands[0]
            let b = hands[1]
            if a.isFist && b.isFist,
               a.handCenterY > 0.35 && a.handCenterY < 0.7,
               b.handCenterY > 0.35 && b.handCenterY < 0.7 {
                candidates.append(("LOVE", 0.72))
            }
        }
        // LOVE (ILY) — thumb+index+pinky
        if primary.thumbExtended && primary.indexExtended && primary.littleExtended
            && !primary.middleExtended && !primary.ringExtended {
            candidates.append(("LOVE", 0.8))
        }

        // HOW — two fists together, thumbs / knuckles roll (two fists close)
        if hands.count >= 2 {
            let a = hands[0]
            let b = hands[1]
            if a.isFist && b.isFist {
                let dx = abs(a.handCenterX - b.handCenterX)
                if dx < 0.2 {
                    candidates.append(("HOW", 0.62))
                }
            }
        }

        // YOU — index pointing toward camera (index only, hand forward-ish)
        if primary.indexExtended && !primary.middleExtended && !primary.ringExtended
            && !primary.littleExtended && primary.handCenterY > 0.35 {
            candidates.append(("YOU", 0.7))
        }

        // ME — index pointing to self (index only, higher / toward signer — hard on rear cam)
        if primary.indexExtended && !primary.middleExtended && primary.handCenterY > 0.55 {
            candidates.append(("ME", 0.55))
        }

        // GOOD — flat hand from chin downward (similar thanks)
        if (primary.isOpenPalm || primary.nonThumbExtendedCount >= 3)
            && primary.handCenterY > 0.55 && motion.deltaY < -0.02 {
            candidates.append(("GOOD", 0.66))
        }

        // BAD — flat hand from mouth turning out/down
        if primary.isOpenPalm && primary.handCenterY > 0.5 && motion.deltaY < -0.03 && abs(motion.deltaX) > 0.02 {
            candidates.append(("BAD", 0.58))
        }

        // MORE — fingertips together tapping (pinch / tips clustered + short motion)
        if primary.tipsNearPalm || primary.thumbIndexPinch {
            if hypot(motion.deltaX, motion.deltaY) > 0.01 {
                candidates.append(("MORE", 0.64))
            }
        }

        // SORRY — fist circling chest
        if primary.isFist && primary.handCenterY > 0.3 && primary.handCenterY < 0.65 {
            if hypot(motion.deltaX, motion.deltaY) > 0.025 {
                candidates.append(("SORRY", 0.6))
            }
        }

        // WHAT — open hands / shrug (open palm + lateral)
        if primary.isOpenPalm && abs(motion.deltaX) > 0.035 {
            candidates.append(("WHAT", 0.55))
        }

        // WHERE — index finger wagging side to side
        if primary.indexExtended && !primary.middleExtended && abs(motion.deltaX) > 0.03 {
            candidates.append(("WHERE", 0.66))
        }

        // OK — F handshape held (thumb-index circle)
        if primary.thumbIndexPinch && primary.middleExtended && primary.ringExtended {
            candidates.append(("OK", 0.74))
        }

        // YES already; STOP — open palm toward camera held
        if primary.isOpenPalm && abs(motion.deltaX) < 0.01 && abs(motion.deltaY) < 0.01
            && primary.handCenterY > 0.4 && primary.handCenterY < 0.75 {
            candidates.append(("STOP", 0.5))
        }

        guard let best = candidates.max(by: { $0.1 < $1.1 }), best.1 >= 0.55 else {
            return nil
        }

        return RecognitionResult(
            label: best.0,
            kind: .everydaySign,
            confidence: min(best.1, 0.95),
            timestamp: Date()
        )
    }
}
