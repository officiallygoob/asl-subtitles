import SwiftUI

/// FaceTime / video-call coach + controls (Liquid Glass chrome).
struct CallModePanel: View {
    @ObservedObject var call: CallModeController
    @ObservedObject var sharePlay: SharePlayCoordinator
    var captionPreview: String
    var onStart: () -> Void
    var onStop: () -> Void
    var onPiP: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Label("Call Mode", systemImage: "video.fill")
                    .font(.headline.weight(.semibold))
                    .foregroundStyle(AppTheme.accent)
                Spacer()
                Text(call.captureSource.rawValue == "none" ? "Idle" : call.captureSource.rawValue)
                    .font(.caption2.monospaced())
                    .foregroundStyle(AppTheme.captionMuted)
            }

            Text(call.statusMessage)
                .font(.subheadline)
                .foregroundStyle(AppTheme.captionSecondary)
                .fixedSize(horizontal: false, vertical: true)

            if call.showCoachMarks {
                coachMarks
            }

            if !captionPreview.isEmpty {
                Text(captionPreview)
                    .font(.title3.weight(.bold))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
                    .background(AppTheme.panelElevated, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            }

            HStack(spacing: 10) {
                Button(action: onStart) {
                    Label("Use with FaceTime", systemImage: "face.smiling.inverse")
                        .font(.subheadline.weight(.semibold))
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                }
                .buttonStyle(.plain)
                .liquidGlassCapsule(.interactive)

                BroadcastPickerView()
                    .frame(width: 52, height: 52)
                    .liquidGlass(.regular, in: Circle())
                    .accessibilityLabel("Start screen broadcast")

                Button(action: onPiP) {
                    Image(systemName: "pip.enter")
                        .padding(12)
                }
                .buttonStyle(.plain)
                .liquidGlass(.interactive, in: Circle())
                .accessibilityLabel("Picture in Picture captions")

                if call.isCallModeActive {
                    Button(action: onStop) {
                        Image(systemName: "stop.fill")
                            .foregroundStyle(AppTheme.danger)
                            .padding(12)
                    }
                    .buttonStyle(.plain)
                    .liquidGlass(.regular, in: Circle())
                }
            }

            Button {
                Task { await sharePlay.startSharing() }
            } label: {
                Label(sharePlay.isSessionActive ? "SharePlay active" : "SharePlay on FaceTime", systemImage: "shareplay")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(AppTheme.signingRole)
            }
            .buttonStyle(.plain)

            if let err = call.lastError {
                Text(err).font(.caption2).foregroundStyle(AppTheme.danger)
            }
            if !sharePlay.status.isEmpty {
                Text(sharePlay.status).font(.caption2).foregroundStyle(AppTheme.captionMuted)
            }
        }
        .padding(16)
        .liquidGlassCard(cornerRadius: 22)
    }

    private var coachMarks: some View {
        VStack(alignment: .leading, spacing: 8) {
            step(1, "Open FaceTime and start your call")
            step(2, "Return here → tap Use with FaceTime (Screen Capture) or the Broadcast button")
            step(3, "Keep ASL Subtitles open or pop PiP captions over FaceTime")
            Text("ASL Subtitles cannot draw inside FaceTime’s own UI — capture + overlay is the supported path.")
                .font(.caption2)
                .foregroundStyle(AppTheme.captionMuted)
        }
        .padding(12)
        .background(AppTheme.panel.opacity(0.9), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    private func step(_ n: Int, _ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text("\(n)")
                .font(.caption.weight(.bold))
                .foregroundStyle(.black)
                .frame(width: 20, height: 20)
                .background(AppTheme.accent, in: Circle())
            Text(text)
                .font(.caption)
                .foregroundStyle(AppTheme.captionSecondary)
        }
    }
}
