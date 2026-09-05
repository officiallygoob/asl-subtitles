import Foundation
import Vision

/// Heuristic recognizer for an expanded everyday ASL vocabulary.
///
/// Signs that need motion use simple frame-to-frame deltas supplied by the caller.
/// Many glosses are *approximate* handshape+motion templates — not fluent ASL.
/// Gloss tokens match server `GLOSS_VOCAB` where possible (uppercase).
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
        let mag = hypot(motion.deltaX, motion.deltaY)
        let two = hands.count >= 2

        // HELLO / HI — open palm near head height, small side motion / wave
        if primary.isOpenPalm && primary.handCenterY > 0.55 {
            let waveBoost = motion.lateralOscillations >= 1 || abs(motion.deltaX) > 0.03
            candidates.append(("HELLO", waveBoost ? 0.8 : 0.62))
            candidates.append(("HI", waveBoost ? 0.72 : 0.55))
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
        if primary.thumbIndexPinch && abs(motion.deltaX) > 0.03 {
            candidates.append(("NO", 0.6))
        }

        // PLEASE — open palm on chest circling
        if primary.isOpenPalm && primary.handCenterY > 0.3 && primary.handCenterY < 0.65 {
            if mag > 0.02 {
                candidates.append(("PLEASE", 0.65))
            }
        }

        // HELP — fist on open palm (two hands) lifting
        if two {
            let a = hands[0]
            let b = hands[1]
            if (a.isFist && b.isOpenPalm) || (b.isFist && a.isOpenPalm) {
                candidates.append(("HELP", motion.deltaY > 0.02 ? 0.78 : 0.6))
            }
        }

        // NAME — H / U handshape tapping
        if primary.indexExtended && primary.middleExtended && !primary.ringExtended && !primary.littleExtended {
            if mag > 0.015 {
                candidates.append(("NAME", 0.58))
            }
        }

        // FRIEND — interlocking index hooks
        if two {
            let a = hands[0]
            let b = hands[1]
            if a.indexExtended && b.indexExtended && !a.middleExtended && !b.middleExtended {
                candidates.append(("FRIEND", 0.7))
            }
        }

        // LOVE — crossed fists or ILY
        if two {
            let a = hands[0]
            let b = hands[1]
            if a.isFist && b.isFist,
               a.handCenterY > 0.35 && a.handCenterY < 0.7,
               b.handCenterY > 0.35 && b.handCenterY < 0.7 {
                candidates.append(("LOVE", 0.72))
            }
        }
        if primary.thumbExtended && primary.indexExtended && primary.littleExtended
            && !primary.middleExtended && !primary.ringExtended {
            candidates.append(("LOVE", 0.8))
        }

        // HOW — two fists together
        if two {
            let a = hands[0]
            let b = hands[1]
            if a.isFist && b.isFist {
                let dx = abs(a.handCenterX - b.handCenterX)
                if dx < 0.2 {
                    candidates.append(("HOW", 0.62))
                }
            }
        }

        // YOU — index pointing toward camera
        if primary.indexExtended && !primary.middleExtended && !primary.ringExtended
            && !primary.littleExtended && primary.handCenterY > 0.35 && primary.handCenterY < 0.55 {
            candidates.append(("YOU", 0.7))
        }

        // ME — index pointing to self (higher)
        if primary.indexExtended && !primary.middleExtended && primary.handCenterY > 0.55 {
            candidates.append(("ME", 0.55))
        }

        // WE / THEY — index with lateral sweep
        if primary.indexExtended && !primary.middleExtended && abs(motion.deltaX) > 0.04 {
            if primary.handCenterY > 0.4 {
                candidates.append(("WE", 0.58))
                candidates.append(("THEY", 0.56))
            }
        }

        // MY — flat hand on chest (still-ish mid torso)
        if (primary.isOpenPalm || primary.nonThumbExtendedCount >= 3)
            && primary.handCenterY > 0.35 && primary.handCenterY < 0.55
            && mag < 0.02 {
            candidates.append(("MY", 0.58))
        }

        // YOUR — flat hand pushing out
        if (primary.isOpenPalm || primary.nonThumbExtendedCount >= 3)
            && motion.deltaY < -0.02 && primary.handCenterY > 0.35 && primary.handCenterY < 0.6 {
            candidates.append(("YOUR", 0.56))
        }

        // GOOD — flat hand from chin downward
        if (primary.isOpenPalm || primary.nonThumbExtendedCount >= 3)
            && primary.handCenterY > 0.55 && motion.deltaY < -0.02 {
            candidates.append(("GOOD", 0.66))
        }

        // BAD — flat hand from mouth turning out/down
        if primary.isOpenPalm && primary.handCenterY > 0.5 && motion.deltaY < -0.03 && abs(motion.deltaX) > 0.02 {
            candidates.append(("BAD", 0.58))
        }

        // FINE — open hand mid chest, small motion
        if primary.isOpenPalm && primary.handCenterY > 0.4 && primary.handCenterY < 0.6 && mag > 0.01 && mag < 0.04 {
            candidates.append(("FINE", 0.55))
        }

        // GREAT — open hand rising
        if primary.isOpenPalm && motion.deltaY > 0.035 {
            candidates.append(("GREAT", 0.6))
        }

        // MORE — fingertips together tapping
        if primary.tipsNearPalm || primary.thumbIndexPinch {
            if mag > 0.01 {
                candidates.append(("MORE", 0.64))
            }
        }

        // LESS — flat hands lowering (two hands)
        if two && hands[0].isOpenPalm && hands[1].isOpenPalm && motion.deltaY < -0.03 {
            candidates.append(("LESS", 0.58))
        }

        // SAME — two indexes together lateral
        if two && hands[0].indexExtended && hands[1].indexExtended && abs(motion.deltaX) > 0.02 {
            candidates.append(("SAME", 0.55))
        }

        // DIFFERENT — indexes crossing apart
        if two && hands[0].indexExtended && hands[1].indexExtended && mag > 0.03 {
            candidates.append(("DIFFERENT", 0.55))
        }

        // SORRY — fist circling chest
        if primary.isFist && primary.handCenterY > 0.3 && primary.handCenterY < 0.65 {
            if mag > 0.025 {
                candidates.append(("SORRY", 0.6))
            }
        }

        // EXCUSE — flat hand brushing lateral mid
        if (primary.isOpenPalm || primary.nonThumbExtendedCount >= 3)
            && abs(motion.deltaX) > 0.04 && primary.handCenterY > 0.4 && primary.handCenterY < 0.6 {
            candidates.append(("EXCUSE", 0.55))
        }

        // WHAT — open hands / shrug
        if primary.isOpenPalm && abs(motion.deltaX) > 0.035 {
            candidates.append(("WHAT", 0.55))
        }

        // WHERE — index finger wagging
        if primary.indexExtended && !primary.middleExtended && abs(motion.deltaX) > 0.03 {
            candidates.append(("WHERE", 0.66))
        }

        // WHEN — index circling (use oscillation / circular mag)
        if primary.indexExtended && !primary.middleExtended && motion.lateralOscillations >= 1 {
            candidates.append(("WHEN", 0.58))
        }

        // WHO — index near face circling
        if primary.indexExtended && !primary.middleExtended && primary.handCenterY > 0.55 && mag > 0.02 {
            candidates.append(("WHO", 0.55))
        }

        // WHY — Y handshape (thumb+pinky) near head
        if primary.thumbExtended && primary.littleExtended && !primary.indexExtended
            && !primary.middleExtended && !primary.ringExtended && primary.handCenterY > 0.5 {
            candidates.append(("WHY", 0.62))
        }

        // WHICH — fist alternating (two fists vertical motion)
        if two && hands[0].isFist && hands[1].isFist && abs(motion.deltaY) > 0.03 {
            candidates.append(("WHICH", 0.55))
        }

        // OK — F handshape held
        if primary.thumbIndexPinch && primary.middleExtended && primary.ringExtended {
            candidates.append(("OK", 0.74))
        }

        // MAYBE — two open palms rocking
        if two && hands[0].isOpenPalm && hands[1].isOpenPalm && abs(motion.deltaY) > 0.025 {
            candidates.append(("MAYBE", 0.58))
        }

        // TRUE — index from chin forward
        if primary.indexExtended && !primary.middleExtended && motion.deltaY < -0.02 && primary.handCenterY > 0.5 {
            candidates.append(("TRUE", 0.55))
        }

        // FALSE — index lateral near face
        if primary.indexExtended && !primary.middleExtended && abs(motion.deltaX) > 0.035 && primary.handCenterY > 0.55 {
            candidates.append(("FALSE", 0.55))
        }

        // STOP — open palm held
        if primary.isOpenPalm && abs(motion.deltaX) < 0.01 && abs(motion.deltaY) < 0.01
            && primary.handCenterY > 0.4 && primary.handCenterY < 0.75 {
            candidates.append(("STOP", 0.5))
        }

        // WAIT — open fingers wiggle (open palm + small oscillations)
        if primary.isOpenPalm && motion.lateralOscillations >= 1 && primary.handCenterY > 0.4 {
            candidates.append(("WAIT", 0.58))
        }

        // WANT — claw pulling toward body (tips curled + toward)
        if primary.nonThumbExtendedCount >= 3 && !primary.isOpenPalm && motion.deltaY > 0.02 {
            candidates.append(("WANT", 0.6))
        }

        // NEED — bent index / X-like nodding down
        if primary.indexExtended && !primary.middleExtended && motion.deltaY < -0.03 {
            candidates.append(("NEED", 0.55))
        }

        // UNDERSTAND — index flick at forehead
        if primary.indexExtended && !primary.middleExtended && primary.handCenterY > 0.6 && motion.deltaY > 0.02 {
            candidates.append(("UNDERSTAND", 0.58))
        }

        // KNOW — flat tips at forehead
        if (primary.isOpenPalm || primary.nonThumbExtendedCount >= 3) && primary.handCenterY > 0.62 && mag < 0.025 {
            candidates.append(("KNOW", 0.56))
        }

        // DONT-KNOW — open hands flip from head
        if two && hands[0].isOpenPalm && hands[1].isOpenPalm && primary.handCenterY > 0.55 && mag > 0.03 {
            candidates.append(("DONT-KNOW", 0.6))
        }

        // LIKE — thumb+middle from chest (approx pinch mid)
        if primary.thumbExtended && primary.middleExtended && !primary.indexExtended
            && primary.handCenterY > 0.35 && primary.handCenterY < 0.55 {
            candidates.append(("LIKE", 0.58))
        }

        // GO — index outward
        if primary.indexExtended && !primary.middleExtended && motion.deltaY < -0.03 {
            candidates.append(("GO", 0.58))
        }

        // COME — index beckoning (toward + up-ish)
        if primary.indexExtended && !primary.middleExtended && motion.deltaY > 0.025 {
            candidates.append(("COME", 0.58))
        }

        // AGAIN — bent hand short tap motion mid
        if primary.nonThumbExtendedCount >= 2 && primary.nonThumbExtendedCount <= 3 && mag > 0.02
            && primary.handCenterY > 0.4 && primary.handCenterY < 0.6 {
            candidates.append(("AGAIN", 0.55))
        }

        // SLOW — flat hand sliding slowly (small mag open palm)
        if primary.isOpenPalm && mag > 0.008 && mag < 0.02 && abs(motion.deltaX) > abs(motion.deltaY) {
            candidates.append(("SLOW", 0.55))
        }

        // FAST — quick lateral flick
        if mag > 0.06 {
            candidates.append(("FAST", 0.55))
        }

        // LOOK — V hand from eyes out
        if primary.indexExtended && primary.middleExtended && !primary.ringExtended
            && primary.handCenterY > 0.55 && motion.deltaY < -0.02 {
            candidates.append(("LOOK", 0.6))
        }

        // EAT — pinch toward mouth
        if (primary.thumbIndexPinch || primary.tipsNearPalm) && primary.handCenterY > 0.55 {
            candidates.append(("EAT", 0.62))
        }

        // DRINK — C hand tilting up near mouth
        if primary.nonThumbExtendedCount >= 3 && !primary.isOpenPalm && primary.thumbExtended
            && primary.handCenterY > 0.5 && motion.deltaY > 0.02 {
            candidates.append(("DRINK", 0.6))
        }

        // HUNGRY — claw down chest
        if primary.nonThumbExtendedCount >= 3 && !primary.isOpenPalm
            && primary.handCenterY > 0.35 && primary.handCenterY < 0.6 && motion.deltaY < -0.03 {
            candidates.append(("HUNGRY", 0.58))
        }

        // HAPPY — flat hands brushing up chest
        if primary.isOpenPalm && motion.deltaY > 0.03 && primary.handCenterY > 0.35 && primary.handCenterY < 0.65 {
            candidates.append(("HAPPY", 0.58))
        }

        // SAD — open hands down face
        if primary.isOpenPalm && motion.deltaY < -0.035 && primary.handCenterY > 0.5 {
            candidates.append(("SAD", 0.56))
        }

        // TIRED — fingertips high dropping
        if two && motion.deltaY < -0.03 && primary.handCenterY > 0.5 {
            candidates.append(("TIRED", 0.55))
        }

        // HOT — claw from mouth out
        if primary.nonThumbExtendedCount >= 3 && !primary.isOpenPalm && primary.handCenterY > 0.55 && motion.deltaY < -0.025 {
            candidates.append(("HOT", 0.55))
        }

        // COLD — fists shaking
        if two && hands[0].isFist && hands[1].isFist && (motion.lateralOscillations >= 1 || mag > 0.03) {
            candidates.append(("COLD", 0.6))
        }

        // TIME — index tapping opposite wrist area (point + short motion mid)
        if primary.indexExtended && !primary.middleExtended && mag > 0.015
            && primary.handCenterY > 0.4 && primary.handCenterY < 0.55 {
            candidates.append(("TIME", 0.55))
        }

        // TODAY — short drop mid
        if two && motion.deltaY < -0.025 && primary.handCenterY > 0.4 && primary.handCenterY < 0.55 {
            candidates.append(("TODAY", 0.52))
        }

        // TOMORROW — A/fist near cheek moving forward
        if primary.isFist && primary.handCenterY > 0.55 && motion.deltaY < -0.02 {
            candidates.append(("TOMORROW", 0.55))
        }

        // LATER — L handshape (index+thumb) twist/circle
        if primary.indexExtended && primary.thumbExtended && !primary.middleExtended
            && !primary.ringExtended && mag > 0.02 {
            candidates.append(("LATER", 0.58))
        }

        // SEE — V or point near eyes out
        if primary.indexExtended && primary.middleExtended && !primary.ringExtended && primary.handCenterY > 0.58 {
            candidates.append(("SEE", 0.55))
        }

        // HOME — flat tips to cheek
        if (primary.isOpenPalm || primary.nonThumbExtendedCount >= 3) && primary.handCenterY > 0.58 && mag < 0.02 {
            candidates.append(("HOME", 0.52))
        }

        // WORK — two fists tapping
        if two && hands[0].isFist && hands[1].isFist && mag > 0.02
            && primary.handCenterY > 0.4 && primary.handCenterY < 0.6 {
            candidates.append(("WORK", 0.58))
        }

        // SCHOOL — flat hands clapping (two open + short motion)
        if two && hands[0].isOpenPalm && hands[1].isOpenPalm && mag > 0.02
            && primary.handCenterY > 0.4 && primary.handCenterY < 0.6 {
            candidates.append(("SCHOOL", 0.55))
        }

        // WRITE — pinch lateral writing
        if primary.thumbIndexPinch && abs(motion.deltaX) > 0.02 && primary.handCenterY > 0.4 {
            candidates.append(("WRITE", 0.58))
        }

        // SPELL — point lateral small (fingerspelling motion)
        if primary.indexExtended && !primary.middleExtended && abs(motion.deltaX) > 0.025
            && primary.handCenterY > 0.45 && primary.handCenterY < 0.65 {
            candidates.append(("SPELL", 0.52))
        }

        // Numbers 1–5, 10 (finger counts; held relatively still to avoid fighting letters)
        if mag < 0.02 {
            let n = primary.nonThumbExtendedCount
            if n == 1 && primary.indexExtended && !primary.thumbExtended {
                candidates.append(("ONE", 0.7))
            } else if n == 2 && primary.indexExtended && primary.middleExtended && !primary.thumbExtended {
                candidates.append(("TWO", 0.68))
            } else if primary.thumbExtended && primary.indexExtended && primary.middleExtended
                        && !primary.ringExtended && !primary.littleExtended {
                candidates.append(("THREE", 0.66))
            } else if n == 4 && !primary.thumbExtended {
                candidates.append(("FOUR", 0.66))
            } else if primary.isOpenPalm {
                candidates.append(("FIVE", 0.55))
            }
            // TEN — thumb up shake approx: thumb only + small motion already filtered; allow slight
            if primary.thumbExtended && n == 0 {
                candidates.append(("TEN", 0.6))
            }
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
