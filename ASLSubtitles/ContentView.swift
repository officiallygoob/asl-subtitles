import SwiftUI
import AVFoundation

struct ContentView: View {
    @StateObject private var session = ASLSessionController()
    @State private var showVocabulary = false
    @State private var showSettings = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            switch session.permissionStatus {
            case .notDetermined, .denied, .restricted:
                PermissionView(status: session.permissionStatus) {
                    session.requestPermissionAndStart()
                }
            case .authorized:
                cameraExperience
            @unknown default:
                PermissionView(status: .denied) {
                    session.requestPermissionAndStart()
                }
            }
        }
        .preferredColorScheme(.dark)
        .onAppear {
            session.prepare()
        }
        .onDisappear {
            session.stop()
        }
        .sheet(isPresented: $showVocabulary) {
            VocabularySheet()
        }
        .sheet(isPresented: $showSettings) {
            SettingsSheet(session: session)
        }
    }

    private var cameraExperience: some View {
        ZStack {
            CameraPreviewView(session: session.captureSession)
                .ignoresSafeArea()

            if session.showLandmarkDebug {
                LandmarkOverlay(hands: session.latestHands)
                    .ignoresSafeArea()
                    .allowsHitTesting(false)
            }

            VStack(spacing: 0) {
                topBar
                Spacer()
                SubtitleOverlay(
                    text: session.smoothedSubtitle,
                    confidence: session.subtitleConfidence,
                    isWatching: session.isWatchingEmpty
                )
                .padding(.horizontal, 20)
                .padding(.bottom, 12)

                ControlsBar(
                    isFrontCamera: session.isFrontCamera,
                    showDebug: session.showLandmarkDebug,
                    onFlipCamera: { session.toggleCamera() },
                    onToggleDebug: { session.showLandmarkDebug.toggle() },
                    onShowVocabulary: { showVocabulary = true },
                    onShowSettings: { showSettings = true }
                )
                .padding(.horizontal, 16)
                .padding(.bottom, 28)
            }
        }
    }

    private var topBar: some View {
        HStack {
            Label("ASL Subtitles", systemImage: "hand.raised.fill")
                .font(.headline.weight(.semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(.ultraThinMaterial, in: Capsule())

            Spacer()

            if session.isProcessing {
                HStack(spacing: 6) {
                    Circle()
                        .fill(Color.green)
                        .frame(width: 8, height: 8)
                    Text("Live")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.white)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(.ultraThinMaterial, in: Capsule())
            }
        }
        .padding(.horizontal, 16)
        .padding(.top, 12)
    }
}

#Preview {
    ContentView()
}
