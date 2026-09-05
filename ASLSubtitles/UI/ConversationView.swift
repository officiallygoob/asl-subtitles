import SwiftUI
import UIKit

/// Primary experience: live ASL→English subtitles + speech→text reverse channel.
/// Liquid Glass (iOS 27) for navigation/controls floating above camera + transcript content.
struct ConversationView: View {
    @ObservedObject var session: ASLSessionController
    @State private var showVocabulary = false
    @State private var showSettings = false
    @State private var showCallMode = false
    @State private var transcriptPulse = false
    @Namespace private var glassNS

    var body: some View {
        ZStack {
            AppTheme.canvas.ignoresSafeArea()

            switch session.permissionStatus {
            case .notDetermined, .denied, .restricted:
                PermissionView(status: session.permissionStatus) {
                    session.requestPermissionAndStart()
                }
            case .authorized:
                conversationExperience
            @unknown default:
                PermissionView(status: .denied) {
                    session.requestPermissionAndStart()
                }
            }
        }
        .preferredColorScheme(.dark)
        .onAppear { session.prepare() }
        .sheet(isPresented: $showVocabulary) { VocabularySheet() }
        .sheet(isPresented: $showSettings) { SettingsSheet(session: session) }
        .sheet(isPresented: $session.showHistorySheet) { HistorySheet(session: session) }
        .sheet(isPresented: $showCallMode) {
            NavigationStack {
                ScrollView {
                    CallModePanel(
                        call: session.callMode,
                        sharePlay: session.sharePlay,
                        captionPreview: session.captionDisplayText,
                        onStart: { session.beginCallMode() },
                        onStop: { session.endCallMode() },
                        onPiP: { session.callMode.startPiP() }
                    )
                    .padding()
                }
                .background(AppTheme.panel.ignoresSafeArea())
                .navigationTitle("FaceTime")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Done") { showCallMode = false }
                            .foregroundStyle(AppTheme.accent)
                    }
                }
            }
            .preferredColorScheme(.dark)
            .presentationDetents([.medium, .large])
        }
        .onChange(of: session.persistentSigningTranscript) { _, _ in
            withAnimation(.spring(response: 0.35, dampingFraction: 0.82)) {
                transcriptPulse = true
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                transcriptPulse = false
            }
        }
    }

    private var conversationExperience: some View {
        GeometryReader { geo in
            let cameraHeight = min(geo.size.height * 0.42, 360.0)
            VStack(spacing: 0) {
                cameraHero(height: cameraHeight)
                transcriptHero
                    .padding(.horizontal, 16)
                    .padding(.top, 10)
                    .padding(.bottom, 6)
                conversationPanel
                    .frame(maxHeight: .infinity)
            }
            .safeAreaInset(edge: .bottom, spacing: 0) {
                floatingControls
                    .padding(.horizontal, 14)
                    .padding(.bottom, 10)
            }
        }
    }

    // MARK: - Camera (content — not glass-on-glass)

    private func cameraHero(height: CGFloat) -> some View {
        ZStack {
            CameraPreviewView(session: session.captureSession)
                .clipShape(RoundedRectangle(cornerRadius: 28, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: 28, style: .continuous)
                        .strokeBorder(
                            LinearGradient(
                                colors: [AppTheme.accent.opacity(0.55), .white.opacity(0.12)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 1.2
                        )
                }
                .overlay {
                    // Soft vignette for caption contrast near edges
                    RoundedRectangle(cornerRadius: 28, style: .continuous)
                        .fill(
                            RadialGradient(
                                colors: [.clear, .black.opacity(0.35)],
                                center: .center,
                                startRadius: 40,
                                endRadius: 280
                            )
                        )
                        .allowsHitTesting(false)
                }
                .shadow(color: .black.opacity(0.45), radius: 18, y: 8)

            if session.showLandmarkDebug {
                HolisticLandmarkOverlay(
                    hands: session.latestHands,
                    bodyJoints: session.latestBodyJoints,
                    face: session.latestFaceJoints,
                    nmm: session.showNMMBadges ? session.latestNMM : nil
                )
                .clipShape(RoundedRectangle(cornerRadius: 28, style: .continuous))
                .allowsHitTesting(false)
            }

            VStack {
                topBar
                    .padding(.horizontal, 14)
                    .padding(.top, 12)
                Spacer()
            }
        }
        .frame(height: height)
        .padding(.horizontal, 12)
        .padding(.top, 8)
    }

    // MARK: - Top bar (Liquid Glass controls)

    private var topBar: some View {
        GlassChrome {
            HStack(spacing: 10) {
                HStack(spacing: 8) {
                    Image(systemName: "hands.and.sparkles.fill")
                        .foregroundStyle(AppTheme.accent)
                    Text("Conversation")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.white)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .liquidGlassCapsule(.regular)
                .glassMorphID("titleChip", in: glassNS)

                Spacer(minLength: 8)

                connectionChip
                    .glassMorphID("connectionChip", in: glassNS)
            }
        }
    }

    @ViewBuilder
    private var connectionChip: some View {
        let (label, color, icon): (String, Color, String) = {
            switch session.recognition.state {
            case .connected: return ("Server", AppTheme.speakingRole, "dot.radiowaves.left.and.right")
            case .connecting: return ("Connecting", .yellow, "antenna.radiowaves.left.and.right")
            case .offlineFallback: return ("Offline", .orange, "iphone")
            case .error: return ("Error", AppTheme.danger, "exclamationmark.triangle.fill")
            case .disconnected: return ("Idle", .gray, "pause.circle")
            }
        }()
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.caption.weight(.semibold))
                .foregroundStyle(color)
            Text(label)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.white)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .liquidGlassCapsule(.interactive)
        .accessibilityLabel("Connection: \(label)")
    }

    // MARK: - Signing transcript hero (content-first readability)

    private var transcriptHero: some View {
        let persistent = session.persistentSigningTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        let current = session.currentSigningWord.trimmingCharacters(in: .whitespacesAndNewlines)
        let showCurrentSeparately: Bool = {
            guard !current.isEmpty else { return false }
            let last = persistent.split(separator: " ").last.map(String.init) ?? ""
            return last.compare(current, options: [.caseInsensitive, .diacriticInsensitive]) != .orderedSame
        }()
        let isIdle = persistent.isEmpty && current.isEmpty

        return VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: "captions.bubble.fill")
                    .foregroundStyle(AppTheme.accent)
                Text("Signing")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(AppTheme.accent)
                    .textCase(.uppercase)
                    .tracking(0.6)
                Spacer()
                if !session.partialGloss.isEmpty {
                    Text(session.partialGloss.joined(separator: " · "))
                        .font(.caption2.monospaced())
                        .foregroundStyle(AppTheme.captionMuted)
                        .lineLimit(1)
                }
            }

            Group {
                if isIdle {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Watching for signs…")
                            .font(.largeTitle.weight(.bold))
                            .foregroundStyle(AppTheme.captionSecondary)
                            .minimumScaleFactor(0.7)
                        Text("Face your friend so hands, face, and shoulders stay in frame. Raised brows and head shakes help shape questions and negation.")
                            .font(.subheadline)
                            .foregroundStyle(AppTheme.captionMuted)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .frame(maxWidth: AppTheme.captionMaxWidth, alignment: .leading)
                    .accessibilityLabel("Watching for signs")
                } else {
                    (
                        Text(persistent.isEmpty ? "" : persistent)
                            .font(.largeTitle.weight(.bold))
                            .foregroundStyle(AppTheme.captionPrimary)
                        + Text(showCurrentSeparately ? (persistent.isEmpty ? current + "…" : " " + current + "…") : "")
                            .font(.title2.weight(.semibold).italic())
                            .foregroundStyle(AppTheme.captionSecondary)
                    )
                    .dynamicTypeSize(...DynamicTypeSize.accessibility3)
                    .frame(maxWidth: AppTheme.captionMaxWidth, alignment: .leading)
                    .multilineTextAlignment(.leading)
                    .lineSpacing(4)
                    .minimumScaleFactor(0.65)
                    .scaleEffect(transcriptPulse ? 1.015 : 1.0)
                    .animation(.spring(response: 0.35, dampingFraction: 0.82), value: transcriptPulse)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .fill(AppTheme.panelElevated.opacity(0.92))
                .overlay {
                    RoundedRectangle(cornerRadius: 22, style: .continuous)
                        .strokeBorder(AppTheme.accent.opacity(0.28), lineWidth: 1)
                }
                .shadow(color: AppTheme.accent.opacity(0.12), radius: 16, y: 4)
        }
        .accessibilityLabel("Signing transcript: \(session.captionDisplayText)")
    }

    // MARK: - Conversation history

    private var conversationPanel: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                Text("History")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(AppTheme.captionSecondary)
                Spacer()
                if session.speech.isListening {
                    Label("Mic on", systemImage: "mic.fill")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(AppTheme.speakingRole)
                }
                Button {
                    UIImpactFeedbackGenerator(style: .light).impactOccurred()
                    session.showHistorySheet = true
                } label: {
                    Image(systemName: "clock.arrow.circlepath")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(AppTheme.captionSecondary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Past conversations")

                Button {
                    UIImpactFeedbackGenerator(style: .light).impactOccurred()
                    session.requestConversationSummary()
                } label: {
                    HStack(spacing: 4) {
                        if session.isSummarizing {
                            ProgressView().controlSize(.mini)
                        } else {
                            Image(systemName: "sparkles")
                        }
                        Text("Summarize")
                    }
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(AppTheme.accent)
                }
                .buttonStyle(.plain)
                .disabled(session.history.isEmpty && session.persistentSigningTranscript.isEmpty)

                Button {
                    UIImpactFeedbackGenerator(style: .light).impactOccurred()
                    session.clearHistory()
                } label: {
                    Text("Clear")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(AppTheme.captionMuted)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 10)

            if !session.conversationSummary.isEmpty {
                summaryBanner
                    .padding(.horizontal, 14)
                    .padding(.bottom, 6)
            }

            if !session.suggestedReplies.isEmpty {
                suggestedReplyChips
                    .padding(.horizontal, 14)
                    .padding(.bottom, 8)
            }

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        if session.history.isEmpty && session.speech.partialTranscript.isEmpty {
                            emptyHistory
                        }
                        ForEach(session.history) { turn in
                            turnBubble(turn)
                                .id(turn.id)
                                .transition(.asymmetric(
                                    insertion: .move(edge: .bottom).combined(with: .opacity),
                                    removal: .opacity
                                ))
                        }
                        if !session.speech.partialTranscript.isEmpty {
                            turnBubble(
                                ConversationTurn(
                                    role: .speaking,
                                    text: session.speech.partialTranscript,
                                    isPartial: true
                                )
                            )
                            .id("partial-speech")
                        }
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .padding(.bottom, 72)
                    .animation(.spring(response: 0.4, dampingFraction: 0.85), value: session.history.count)
                }
                .onChange(of: session.history.count) { _, _ in
                    if let last = session.history.last {
                        withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                    }
                }
                .onChange(of: session.speech.partialTranscript) { _, _ in
                    withAnimation { proxy.scrollTo("partial-speech", anchor: .bottom) }
                }
            }
        }
        .background(AppTheme.panel.ignoresSafeArea(edges: .bottom))
    }

    private var emptyHistory: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Conversation will land here")
                .font(.headline)
                .foregroundStyle(AppTheme.captionSecondary)
            Text("Signed English captions and what you say appear as bubbles — clear roles, easy to skim.")
                .font(.subheadline)
                .foregroundStyle(AppTheme.captionMuted)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(AppTheme.panelElevated)
        )
    }

    private func turnBubble(_ turn: ConversationTurn) -> some View {
        let isSigning = turn.role == .signing
        let roleColor = isSigning ? AppTheme.signingRole : AppTheme.speakingRole
        HStack {
            if !isSigning { Spacer(minLength: 36) }
            VStack(alignment: isSigning ? .leading : .trailing, spacing: 6) {
                HStack(spacing: 6) {
                    Image(systemName: isSigning ? "hand.wave.fill" : "waveform")
                        .font(.caption2.weight(.bold))
                    Text(isSigning ? "Signing" : (turn.role == .speaking ? "You said" : "System"))
                        .font(.caption2.weight(.bold))
                }
                .foregroundStyle(roleColor)

                Text(turn.text)
                    .font(.body.weight(.semibold))
                    .dynamicTypeSize(...DynamicTypeSize.accessibility3)
                    .foregroundStyle(.white)
                    .opacity(turn.isPartial ? 0.7 : 1)
                    .multilineTextAlignment(isSigning ? .leading : .trailing)

                if !turn.gloss.isEmpty && isSigning {
                    Text(turn.gloss.joined(separator: " "))
                        .font(.caption2.monospaced())
                        .foregroundStyle(AppTheme.captionMuted)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .background {
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(roleColor.opacity(0.16))
                    .overlay {
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .strokeBorder(roleColor.opacity(0.28), lineWidth: 1)
                    }
            }
            if isSigning { Spacer(minLength: 36) }
        }
    }

    // MARK: - Floating control capsule (Liquid Glass)

    private var floatingControls: some View {
        GlassChrome {
            HStack(spacing: 6) {
                controlButton(icon: "arrow.triangle.2.circlepath.camera", label: "Flip") {
                    UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                    session.toggleCamera()
                }
                .glassMorphID("ctrlFlip", in: glassNS)

                controlButton(
                    icon: session.showLandmarkDebug ? "eye.fill" : "eye",
                    label: "Debug",
                    active: session.showLandmarkDebug
                ) {
                    UIImpactFeedbackGenerator(style: .light).impactOccurred()
                    session.showLandmarkDebug.toggle()
                }
                .glassMorphID("ctrlDebug", in: glassNS)

                controlButton(
                    icon: session.speech.isListening ? "mic.fill" : "mic.slash",
                    label: "Mic",
                    active: session.speech.isListening
                ) {
                    UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                    session.toggleSpeech()
                }
                .glassMorphID("ctrlMic", in: glassNS)

                controlButton(icon: "character.book.closed", label: "Vocab") {
                    UIImpactFeedbackGenerator(style: .light).impactOccurred()
                    showVocabulary = true
                }
                .glassMorphID("ctrlVocab", in: glassNS)

                controlButton(icon: "gearshape", label: "Settings") {
                    UIImpactFeedbackGenerator(style: .light).impactOccurred()
                    showSettings = true
                }
                .glassMorphID("ctrlSettings", in: glassNS)
            }
            .padding(6)
            .liquidGlass(.interactive, in: Capsule())
        }
        .shadow(color: .black.opacity(0.35), radius: 16, y: 6)
    }

    private func controlButton(
        icon: String,
        label: String,
        active: Bool = false,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            VStack(spacing: 2) {
                Image(systemName: icon)
                    .font(.body.weight(.semibold))
                    .symbolRenderingMode(.hierarchical)
                    .foregroundStyle(active ? AppTheme.accent : .white)
                    .frame(width: 44, height: 28)
                Text(label)
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(active ? AppTheme.accent : AppTheme.captionMuted)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }

    private var summaryBanner: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "sparkles")
                .foregroundStyle(AppTheme.accent)
            VStack(alignment: .leading, spacing: 4) {
                Text("Summary")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(AppTheme.accent)
                    .textCase(.uppercase)
                Text(session.conversationSummary)
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(AppTheme.captionPrimary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
            Button {
                session.conversationSummary = ""
            } label: {
                Image(systemName: "xmark")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(AppTheme.captionMuted)
            }
            .buttonStyle(.plain)
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(AppTheme.accent.opacity(0.12))
                .overlay {
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .strokeBorder(AppTheme.accent.opacity(0.25), lineWidth: 1)
                }
        )
    }

    private var suggestedReplyChips: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Suggested replies")
                .font(.caption2.weight(.bold))
                .foregroundStyle(AppTheme.captionMuted)
                .textCase(.uppercase)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(session.suggestedReplies, id: \.self) { reply in
                        Button {
                            session.applySuggestedReply(reply)
                        } label: {
                            Text(reply)
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(.white)
                                .padding(.horizontal, 14)
                                .padding(.vertical, 8)
                                .liquidGlassCapsule(.interactive)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }
}

