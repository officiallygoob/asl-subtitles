import Foundation
import LiveKit

/// On-device bridge from a remote LiveKit video track into holistic pose landmarks.
///
/// Pipeline (pixels never leave the phone):
/// 1. LiveKit calls `render(frame:)` with a `VideoFrame`.
/// 2. `VideoFrame.toCMSampleBuffer()` produces a `CMSampleBuffer`.
/// 3. `HolisticPoseDetector.detect(in:isFrontCamera:)` extracts landmarks (`isFrontCamera` is
///    false — this is the friend's camera, not a mirrored selfie).
/// 4. The `LandmarkFrame` is delivered to `ASLSessionController.noteRemoteCallLandmarks`.
///
/// TODO(call-captions): `noteRemoteCallLandmarks` does not yet call `processFrame`, so the
/// existing buffer → `SignRecognizer` caption path is not driven by remote video. When that
/// is wired, delete this type's private detector and pass the `CMSampleBuffer` straight into
/// `ASLSessionController.processFrame` so Vision runs once.
final class RemoteSignFrameBridge: NSObject, VideoRenderer, @unchecked Sendable {
    private let ingestor = RemoteSignIngestor()
    private var attachedObject: Track?
    private var detachRenderer: (@MainActor () -> Void)?

    /// Main-actor callback installed by `InAppCallController`.
    var onLandmarks: ((LandmarkFrame) -> Void)? {
        get { ingestor.onLandmarks }
        set { ingestor.onLandmarks = newValue }
    }

    @MainActor var isAdaptiveStreamEnabled: Bool { false }
    @MainActor var adaptiveStreamSize: CGSize { .zero }

    @MainActor
    func attach(_ track: VideoTrack) {
        let object = track as Track
        if let attachedObject, attachedObject === object { return }
        detach()
        attachedObject = object
        track.add(videoRenderer: self)
        detachRenderer = { [weak self] in
            guard let self else { return }
            track.remove(videoRenderer: self)
        }
    }

    @MainActor
    func detach() {
        detachRenderer?()
        detachRenderer = nil
        attachedObject = nil
    }

    nonisolated func render(frame: VideoFrame) {
        ingestor.submit(frame)
    }
}

/// Background ingest so `render(frame:)` does not run Vision on the WebRTC callback thread.
private final class RemoteSignIngestor: @unchecked Sendable {
    private let queue = DispatchQueue(label: "com.aslsubtitles.call.remote-sign", qos: .userInitiated)
    private let detector = HolisticPoseDetector()
    private let callbackLock = NSLock()
    private var callback: ((LandmarkFrame) -> Void)?
    private var seen = 0
    private var busy = false

    var onLandmarks: ((LandmarkFrame) -> Void)? {
        get {
            callbackLock.lock()
            defer { callbackLock.unlock() }
            return callback
        }
        set {
            callbackLock.lock()
            callback = newValue
            callbackLock.unlock()
        }
    }

    func submit(_ frame: VideoFrame) {
        queue.async { [weak self] in
            self?.process(frame)
        }
    }

    private func process(_ frame: VideoFrame) {
        seen += 1
        // ~10 fps from a 30 fps stream. Skip if the previous detect is still running.
        guard seen % 3 == 0, !busy else { return }
        guard let sample = frame.toCMSampleBuffer() else { return }
        busy = true
        defer { busy = false }
        let landmarks = detector.detect(in: sample, isFrontCamera: false)
        callbackLock.lock()
        let handler = callback
        callbackLock.unlock()
        guard let handler else { return }
        DispatchQueue.main.async {
            handler(landmarks)
        }
    }
}
