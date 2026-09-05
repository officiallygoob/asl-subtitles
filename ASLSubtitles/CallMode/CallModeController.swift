import AVKit
import Combine
import Foundation
import SwiftUI

#if canImport(ReplayKit)
import ReplayKit
#endif

#if canImport(ScreenCaptureKit)
import ScreenCaptureKit
#endif

/// Orchestrates FaceTime / video-call captioning:
/// ScreenCaptureKit (preferred) + ReplayKit broadcast fallback + PiP + SharePlay hook.
@MainActor
final class CallModeController: NSObject, ObservableObject {
    enum CaptureSource: String {
        case none
        case screenCaptureKit
        case replayKitBroadcast
        case deviceCamera
    }

    @Published var isCallModeActive = false
    @Published var captureSource: CaptureSource = .none
    @Published var statusMessage: String = "Ready for FaceTime"
    @Published var showCoachMarks = false
    @Published var isPiPActive = false
    @Published var sharePlayActive = false
    @Published var lastError: String?

    /// Frames delivered to the session Vision pipeline (BGRA / sample buffer bridge).
    var onSampleBuffer: ((CMSampleBuffer) -> Void)?

    private var sckStream: AnyObject?
    private var pipController: AVPictureInPictureController?
    private(set) var pipVideoCallViewController: AVPictureInPictureVideoCallViewController?

    // App Group for Broadcast Upload → main app captions
    static let appGroupID = "group.com.officiallygoob.aslsubtitles"
    static let broadcastCaptionKey = "broadcast.liveCaption"
    static let broadcastActiveKey = "broadcast.active"

    func startCallMode() {
        isCallModeActive = true
        showCoachMarks = true
        statusMessage = "Open FaceTime, then start screen capture"
        lastError = nil
        // Prefer SCK; fall back messaging guides user to Broadcast.
        Task { await startScreenCaptureKitIfPossible() }
    }

    func stopCallMode() {
        isCallModeActive = false
        captureSource = .none
        statusMessage = "Call Mode off"
        stopScreenCaptureKit()
        stopPiP()
    }

    // MARK: - ScreenCaptureKit (iOS 27+)

    func startScreenCaptureKitIfPossible() async {
        #if canImport(ScreenCaptureKit)
        if #available(iOS 27.0, *) {
            do {
                let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
                // Prefer FaceTime app window when present; else full display.
                let faceTimeApp = content.applications.first {
                    let name = $0.applicationName.lowercased()
                    return name.contains("facetime")
                }
                let filter: SCContentFilter
                if let faceTimeApp,
                   let window = content.windows.first(where: { $0.owningApplication?.bundleIdentifier == faceTimeApp.bundleIdentifier }) {
                    filter = SCContentFilter(desktopIndependentWindow: window)
                    statusMessage = "Capturing FaceTime window"
                } else if let display = content.displays.first {
                    filter = SCContentFilter(display: display, excludingApplications: [], exceptingWindows: [])
                    statusMessage = "Capturing display — open FaceTime"
                } else {
                    statusMessage = "No display to capture — use Broadcast instead"
                    return
                }

                let config = SCStreamConfiguration()
                config.width = 1280
                config.height = 720
                config.minimumFrameInterval = CMTime(value: 1, timescale: 12)
                config.queueDepth = 4
                config.showsCursor = false
                config.capturesAudio = false

                let stream = SCStream(filter: filter, configuration: config, delegate: nil)
                let output = ASLCaptureOutput(owner: self)
                try stream.addStreamOutput(output, type: .screen, sampleHandlerQueue: DispatchQueue(label: "asl.sck"))
                try await stream.startCapture()
                sckStream = stream
                captureSource = .screenCaptureKit
                showCoachMarks = false
                return
            } catch {
                lastError = error.localizedDescription
                statusMessage = "Screen capture unavailable — try Broadcast from Control Center"
            }
        }
        #endif
        statusMessage = "Use Control Center → Screen Broadcast → ASL Subtitles"
    }

    func stopScreenCaptureKit() {
        #if canImport(ScreenCaptureKit)
        if #available(iOS 27.0, *), let stream = sckStream as? SCStream {
            Task { try? await stream.stopCapture() }
        }
        #endif
        sckStream = nil
        if captureSource == .screenCaptureKit { captureSource = .none }
    }

    fileprivate func handleCapturedBuffer(_ sampleBuffer: CMSampleBuffer) {
        onSampleBuffer?(sampleBuffer)
    }

    // MARK: - ReplayKit broadcast picker helper

    /// Embed `RPSystemBroadcastPickerView` in SwiftUI for “Broadcast ASL Subtitles”.
    func preferredBroadcastExtensionBundleID() -> String {
        "com.officiallygoob.aslsubtitles.broadcast"
    }

    func pollBroadcastCaptions() -> String? {
        guard let defaults = UserDefaults(suiteName: Self.appGroupID) else { return nil }
        let active = defaults.bool(forKey: Self.broadcastActiveKey)
        if active {
            captureSource = .replayKitBroadcast
            statusMessage = "Receiving FaceTime broadcast"
        }
        return defaults.string(forKey: Self.broadcastCaptionKey)
    }

    // MARK: - PiP (video-call style)

    func configurePiP(sourceView: UIView?) {
        // AVPictureInPictureVideoCallViewController keeps a caption surface over FaceTime when possible.
        let vc = AVPictureInPictureVideoCallViewController()
        vc.preferredContentSize = CGSize(width: 360, height: 160)
        pipVideoCallViewController = vc
        // Content is hosted by CallModePiPCaptionView via UIViewControllerRepresentable.
    }

    func startPiP() {
        guard let pipController, pipController.isPictureInPicturePossible else {
            statusMessage = "PiP not available yet — start Call Mode first"
            return
        }
        pipController.startPictureInPicture()
        isPiPActive = true
    }

    func stopPiP() {
        pipController?.stopPictureInPicture()
        isPiPActive = false
    }

    func attachPiPController(_ controller: AVPictureInPictureController) {
        pipController = controller
        pipController?.delegate = self
    }
}

extension CallModeController: AVPictureInPictureControllerDelegate {
    nonisolated func pictureInPictureControllerDidStartPictureInPicture(_ pictureInPictureController: AVPictureInPictureController) {
        Task { @MainActor in self.isPiPActive = true }
    }

    nonisolated func pictureInPictureControllerDidStopPictureInPicture(_ pictureInPictureController: AVPictureInPictureController) {
        Task { @MainActor in self.isPiPActive = false }
    }
}

#if canImport(ScreenCaptureKit)
@available(iOS 27.0, *)
final class ASLCaptureOutput: NSObject, SCStreamOutput {
    weak var owner: CallModeController?
    init(owner: CallModeController) { self.owner = owner }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, sampleBuffer.isValid else { return }
        Task { @MainActor in
            self.owner?.handleCapturedBuffer(sampleBuffer)
        }
    }
}
#endif
