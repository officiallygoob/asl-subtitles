import SwiftUI

/// Accessibility-first Conversation Mode palette + Liquid Glass helpers (iOS 27+).
enum AppTheme {
    /// Accent that stays readable over camera video (cyan-mint).
    static let accent = Color(red: 0.35, green: 0.92, blue: 0.88)
    static let accentSoft = Color(red: 0.35, green: 0.92, blue: 0.88).opacity(0.22)
    static let signingRole = Color(red: 0.45, green: 0.85, blue: 1.0)
    static let speakingRole = Color(red: 0.45, green: 0.95, blue: 0.65)
    static let danger = Color(red: 1.0, green: 0.40, blue: 0.45)
    static let canvas = Color.black
    static let panel = Color(red: 0.07, green: 0.08, blue: 0.10)
    static let panelElevated = Color(red: 0.11, green: 0.12, blue: 0.15)

    static let captionPrimary = Color.white
    static let captionSecondary = Color.white.opacity(0.72)
    static let captionMuted = Color.white.opacity(0.45)

    /// Ideal caption line length guidance (~40–60 ch feels readable).
    static let captionMaxWidth: CGFloat = 560
}

/// Applies Liquid Glass when running on iOS 27+, with solid material fallback.
struct LiquidGlassBackground<S: Shape>: ViewModifier {
    enum Style {
        case regular
        case clear
        case interactive
    }

    var style: Style = .regular
    var shape: S

    func body(content: Content) -> some View {
        if #available(iOS 27.0, *) {
            switch style {
            case .regular:
                content.glassEffect(.regular, in: shape)
            case .clear:
                content.glassEffect(.clear, in: shape)
            case .interactive:
                content.glassEffect(.regular.interactive(), in: shape)
            }
        } else {
            content
                .background(.ultraThinMaterial, in: shape)
        }
    }
}

extension View {
    func liquidGlass<S: Shape>(_ style: LiquidGlassBackground<S>.Style = .regular, in shape: S) -> some View {
        modifier(LiquidGlassBackground(style: style, shape: shape))
    }

    func liquidGlassCapsule(_ style: LiquidGlassBackground<Capsule>.Style = .interactive) -> some View {
        modifier(LiquidGlassBackground(style: style, shape: Capsule()))
    }

    func liquidGlassCard(cornerRadius: CGFloat = 20) -> some View {
        modifier(LiquidGlassBackground(style: .regular, shape: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)))
    }
}


/// Groups glass controls for morphing / performance (iOS 27 Liquid Glass).
struct GlassChrome<Content: View>: View {
    @ViewBuilder var content: () -> Content
    var body: some View {
        if #available(iOS 27.0, *) {
            GlassEffectContainer { content() }
        } else {
            content()
        }
    }
}

extension View {
    @ViewBuilder
    func glassMorphID(_ id: String, in ns: Namespace.ID) -> some View {
        if #available(iOS 27.0, *) {
            self.glassEffectID(id, in: ns)
        } else {
            self
        }
    }
}
