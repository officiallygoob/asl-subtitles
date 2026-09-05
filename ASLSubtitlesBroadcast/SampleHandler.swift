import ReplayKit
import Vision
import UIKit

/// Broadcast Upload Extension: capture FaceTime (via screen broadcast) and write
/// lightweight caption hints into the App Group for the main app / PiP.
/// Full holistic recognition runs best in the main app via ScreenCaptureKit;
/// this extension keeps a heartbeat + optional coarse activity signal while FaceTime is frontmost.
class SampleHandler: RPBroadcastSampleHandler {
    private let appGroup = "group.com.officiallygoob.aslsubtitles"
    private var frameCount = 0

    override func broadcastStarted(withSetupInfo setupInfo: [String: NSObject]?) {
        let defaults = UserDefaults(suiteName: appGroup)
        defaults?.set(true, forKey: "broadcast.active")
        defaults?.set("Broadcast started — open ASL Subtitles for captions", forKey: "broadcast.liveCaption")
        defaults?.synchronize()
    }

    override func broadcastPaused() {
        UserDefaults(suiteName: appGroup)?.set("Broadcast paused", forKey: "broadcast.liveCaption")
    }

    override func broadcastResumed() {
        UserDefaults(suiteName: appGroup)?.set(true, forKey: "broadcast.active")
    }

    override func broadcastFinished() {
        let defaults = UserDefaults(suiteName: appGroup)
        defaults?.set(false, forKey: "broadcast.active")
        defaults?.set("", forKey: "broadcast.liveCaption")
        defaults?.synchronize()
    }

    override func processSampleBuffer(_ sampleBuffer: CMSampleBuffer, with sampleBufferType: RPSampleBufferType) {
        guard sampleBufferType == .video else { return }
        frameCount += 1
        // Keep extension light — every ~15 frames, stamp a heartbeat so the main app knows broadcast is alive.
        if frameCount % 15 == 0 {
            let defaults = UserDefaults(suiteName: appGroup)
            defaults?.set(Date().timeIntervalSince1970, forKey: "broadcast.lastFrame")
            defaults?.set(true, forKey: "broadcast.active")
        }
    }
}
