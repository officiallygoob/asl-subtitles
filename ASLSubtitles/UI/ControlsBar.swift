import SwiftUI

struct ControlsBar: View {
    let isFrontCamera: Bool
    let showDebug: Bool
    let onFlipCamera: () -> Void
    let onToggleDebug: () -> Void
    let onShowVocabulary: () -> Void
    let onShowSettings: () -> Void

    var body: some View {
        HStack(spacing: 14) {
            controlButton(
                icon: "arrow.triangle.2.circlepath.camera",
                label: isFrontCamera ? "Front" : "Rear",
                action: onFlipCamera
            )

            controlButton(
                icon: "text.book.closed",
                label: "Signs",
                action: onShowVocabulary
            )

            controlButton(
                icon: showDebug ? "hand.draw.fill" : "hand.draw",
                label: "Debug",
                emphasized: showDebug,
                action: onToggleDebug
            )

            controlButton(
                icon: "info.circle",
                label: "About",
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
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 18, weight: .semibold))
                Text(label)
                    .font(.caption2.weight(.medium))
            }
            .foregroundStyle(emphasized ? Color.orange : Color.white)
            .frame(width: 64, height: 52)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }
}
