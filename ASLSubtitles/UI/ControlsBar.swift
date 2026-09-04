import SwiftUI

struct ControlsBar: View {
    let isFrontCamera: Bool
    let showDebug: Bool
    var isMicOn: Bool = false
    var serverState: RecognitionConnectionState = .disconnected
    let onFlipCamera: () -> Void
    let onToggleDebug: () -> Void
    var onToggleMic: (() -> Void)? = nil
    let onShowVocabulary: () -> Void
    let onShowSettings: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            controlButton(
                icon: "arrow.triangle.2.circlepath.camera",
                label: isFrontCamera ? "Front" : "Rear",
                action: onFlipCamera
            )

            if let onToggleMic {
                controlButton(
                    icon: isMicOn ? "mic.fill" : "mic.slash",
                    label: isMicOn ? "Mic" : "Mic",
                    emphasized: isMicOn,
                    tint: .green,
                    action: onToggleMic
                )
            }

            controlButton(
                icon: "text.book.closed",
                label: "Signs",
                action: onShowVocabulary
            )

            controlButton(
                icon: showDebug ? "figure.stand" : "figure.stand",
                label: "Debug",
                emphasized: showDebug,
                action: onToggleDebug
            )

            controlButton(
                icon: "gearshape",
                label: "Settings",
                action: onShowSettings
            )
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial, in: Capsule())
    }

    private func controlButton(
        icon: String,
        label: String,
        emphasized: Bool = false,
        tint: Color = .orange,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 17, weight: .semibold))
                Text(label)
                    .font(.caption2.weight(.medium))
            }
            .foregroundStyle(emphasized ? tint : Color.white)
            .frame(width: 58, height: 52)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }
}
