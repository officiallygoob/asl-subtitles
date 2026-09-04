# ASL Subtitles

An iOS SwiftUI starter app that helps **hearing people** follow a **deaf friend’s American Sign Language (ASL)** by showing **live English subtitles** from the camera.

> **Honest framing:** this is a **limited MVP** — fingerspelling A–Z plus ~20 everyday signs using **on-device** Apple Vision hand landmarks and **heuristics**. It is **not** fluent ASL translation and will misread many real-world signs.

Privacy: **all recognition runs on device**. No video or landmarks are uploaded.

## Requirements

- macOS with **Xcode 15+**
- iPhone or iPad running **iOS 17+** (hand pose works best on a **physical device**; Simulator has no real camera/hands)
- Apple Developer signing (Personal Team is fine for device installs)

## Open & run

1. Clone this repo (or unzip `ASL-Subtitles.zip`).
2. Open **`ASLSubtitles.xcodeproj`** in Xcode.
3. Select the **ASLSubtitles** scheme and your connected iPhone.
4. Set your **Team** under *Signing & Capabilities* if prompted (`com.officiallygoob.aslsubtitles`).
5. Build & Run (⌘R). Allow camera access when asked.

### First-run tips

- Prefer **rear camera** with the signer facing you, hand large in frame, good lighting.
- Tap **Signs** to see the supported vocabulary.
- Toggle **Debug** to overlay Vision hand landmarks.
- Front/rear flip is in the bottom control bar.

## What’s included

| Area | Implementation |
|------|----------------|
| Camera | `AVFoundation` preview, rear default, front/rear toggle, `NSCameraUsageDescription` |
| Hand tracking | `VNDetectHumanHandPoseRequest` (Vision) |
| Fingerspelling | Rule/heuristic classifier A–Z from landmarks |
| Everyday signs | ~20 heuristics (hello, thanks, yes, no, please, help, name, friend, love, how, you, me, good, bad, more, sorry, bye, what, where, ok, stop, …) |
| Subtitles | Large high-contrast overlay + temporal majority-vote smoothing |
| UX | Permission screen, “Watching for signs…”, confidence fade, vocabulary sheet |
| Extensibility | `LandmarkFeatures.featureVector()` + TODOs for a future Core ML model |

## Architecture

```
CameraManager (AVCaptureSession)
        ↓ frames
HandPoseDetector (Vision hand pose)
        ↓ HandPoseSnapshot[]
SignRecognizer
   ├─ EverydaySignHeuristics  (multi-hand + motion)
   └─ FingerspellingClassifier (static shapes)
        ↓ RecognitionResult
TemporalSmoother  →  SubtitleOverlay
```

Key folders under `ASLSubtitles/`:

- `Camera/` — capture session + SwiftUI preview
- `Vision/` — hand pose detection + landmark types
- `Recognition/` — features, fingerspelling, everyday signs, orchestrator
- `UI/` — subtitles, controls, permission, vocabulary, debug overlay
- `Utilities/` — temporal smoother, vocabulary catalog

## How to add a new everyday sign

1. Add a display entry in `Utilities/VocabularyCatalog.swift`.
2. Add a heuristic branch in `Recognition/EverydaySignHeuristics.swift` using `LandmarkFeatures` (finger extended flags, pinch, fist, open palm, motion deltas).
3. Keep confidence conservative (`~0.55–0.8`) so the smoother can reject flicker.
4. (Later) Collect labeled clips and replace the branch with a Core ML score — see TODOs in `LandmarkFeatures`, `FingerspellingClassifier`, `EverydaySignHeuristics`, and `SignRecognizer`.

## Limitations (please read)

- **Not fluent ASL** — classifier vocabulary is tiny; grammar, classifiers, facial grammar, and most signs are unsupported.
- **Ambiguous letters** — e.g. M/N/S/T, U/V/H, J/Z (motion) are weak from a single viewpoint.
- **Motion signs** are approximated with short Δx/Δy; they need clearer, slower signing than fluent conversation.
- **Lighting / distance / occlusion** strongly affect Vision confidence.
- **No App Icon image** is bundled (asset slot exists); Xcode may show a default icon until you add a 1024×1024 marketing icon.
- Built on Linux for delivery — **verify the first build on a Mac** and adjust signing.

## License

Created for personal / educational use. ASL is a living language — treat recognition errors with care and never rely on this app for safety-critical communication.
