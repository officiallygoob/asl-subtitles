import SwiftUI

struct SubtitleOverlay: View {
    let text: String
    let confidence: Double
    let isWatching: Bool

    var body: some View {
        Group {
            if isWatching || text.isEmpty {
                watchingState
            } else {
                caption
            }
        }
        .animation(.easeInOut(duration: 0.25), value: text)
        .animation(.easeInOut(duration: 0.25), value: isWatching)
    }

    private var watchingState: some View {
        HStack(spacing: 10) {
            Image(systemName: "eye")
                .font(.title3)
            Text("Watching for signs…")
                .font(.title3.weight(.medium))
        }
        .foregroundStyle(.white.opacity(0.85))
        .padding(.horizontal, 22)
        .padding(.vertical, 16)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(Color.black.opacity(0.45))
                .overlay(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .strokeBorder(Color.white.opacity(0.12), lineWidth: 1)
                )
        )
        .accessibilityLabel("Watching for signs")
    }

    private var caption: some View {
        Text(text)
            .font(.largeTitle.weight(.bold))
            .dynamicTypeSize(...DynamicTypeSize.accessibility3)
            .minimumScaleFactor(0.5)
            .multilineTextAlignment(.center)
            .foregroundStyle(.white)
            .shadow(color: .black.opacity(0.8), radius: 2, x: 0, y: 1)
            .padding(.horizontal, 24)
            .padding(.vertical, 18)
            .frame(maxWidth: .infinity)
            .background(
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .fill(Color.black.opacity(backgroundOpacity))
                    .overlay(
                        RoundedRectangle(cornerRadius: 20, style: .continuous)
                            .strokeBorder(Color.white.opacity(0.18), lineWidth: 1)
                    )
            )
            .opacity(max(0.35, min(1.0, confidence + 0.25)))
            .accessibilityLabel("Subtitle: \(text)")
    }

    private var backgroundOpacity: Double {
        0.55 + min(0.3, confidence * 0.35)
    }
}
