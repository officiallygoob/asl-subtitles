import SwiftUI

/// Primary experience: live ASL→English subtitles + speech→text reverse channel.
struct ConversationView: View {
    @ObservedObject var session: ASLSessionController
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
                conversationExperience
            @unknown default:
                PermissionView(status: .denied) {
                    session.requestPermissionAndStart()
                }
            }
        }
        .preferredColorScheme(.dark)
        .onAppear { session.prepare() }
        .onDisappear { session.stop() }
        .sheet(isPresented: $showVocabulary) { VocabularySheet() }
        .sheet(isPresented: $showSettings) { SettingsSheet(session: session) }
    }

    private var conversationExperience: some View {
        ZStack {
            // Camera fills top ~55%
            VStack(spacing: 0) {
                ZStack {
                    CameraPreviewView(session: session.captureSession)
                    if session.showLandmarkDebug {
                        HolisticLandmarkOverlay(
                            hands: session.latestHands,
                            bodyJoints: session.latestBodyJoints,
                            face: session.latestFaceJoints
                        )
                        .allowsHitTesting(false)
                    }
                    VStack {
                        topBar
                        Spacer()
                        liveSigningCaption
                            .padding(.horizontal, 16)
                            .padding(.bottom, 10)
                    }
                }
                .frame(maxHeight: .infinity)

                conversationPanel
                    .frame(minHeight: 300, idealHeight: 340, maxHeight: 420)
            }
            .ignoresSafeArea(edges: .top)

            VStack {
                Spacer()
                ControlsBar(
                    isFrontCamera: session.isFrontCamera,
                    showDebug: session.showLandmarkDebug,
                    isMicOn: session.speech.isListening,
                    serverState: session.recognition.state,
                    onFlipCamera: { session.toggleCamera() },
                    onToggleDebug: { session.showLandmarkDebug.toggle() },
                    onToggleMic: { session.toggleSpeech() },
                    onShowVocabulary: { showVocabulary = true },
                    onShowSettings: { showSettings = true }
                )
                .padding(.horizontal, 12)
                .padding(.bottom, 18)
            }
        }
    }

    private var topBar: some View {
        HStack {
            Label("Conversation", systemImage: "person.2.wave.2.fill")
                .font(.headline.weight(.semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(.ultraThinMaterial, in: Capsule())

            Spacer()

            connectionChip
        }
        .padding(.horizontal, 16)
        .padding(.top, 12)
    }

    @ViewBuilder
    private var connectionChip: some View {
        let (label, color): (String, Color) = {
            switch session.recognition.state {
            case .connected: return ("Server", .green)
            case .connecting: return ("Connecting", .yellow)
            case .offlineFallback: return ("Offline", .orange)
            case .error: return ("Error", .red)
            case .disconnected: return ("Idle", .gray)
            }
        }()
        HStack(spacing: 6) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(label)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.white)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial, in: Capsule())
    }

    private var liveSigningCaption: some View {
        let persistent = session.persistentSigningTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        let current = session.currentSigningWord.trimmingCharacters(in: .whitespacesAndNewlines)
        let showCurrentSeparately: Bool = {
            guard !current.isEmpty else { return false }
            let last = persistent.split(separator: " ").last.map(String.init) ?? ""
            return last.compare(current, options: [.caseInsensitive, .diacriticInsensitive]) != .orderedSame
        }()
        let isIdle = persistent.isEmpty && current.isEmpty

        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "hands.and.sparkles.fill")
                Text("Signing transcript")
                    .fontWeight(.semibold)
                Spacer()
                if !session.partialGloss.isEmpty {
                    Text(session.partialGloss.joined(separator: " · "))
                        .font(.caption2.monospaced())
                        .foregroundStyle(.white.opacity(0.7))
                        .lineLimit(1)
                }
            }
            .font(.caption)
            .foregroundStyle(.cyan)

            ScrollView {
                VStack(alignment: .leading, spacing: 4) {
                    if isIdle {
                        Text("Watching for signs…")
                            .font(.title2.weight(.bold))
                            .foregroundStyle(.white.opacity(0.55))
                    } else {
                        (
                            Text(persistent.isEmpty ? "" : persistent)
                                .font(.title2.weight(.bold))
                                .foregroundStyle(.white)
                            + Text(showCurrentSeparately ? (persistent.isEmpty ? current + "…" : " " + current + "…") : "")
                                .font(.title2.weight(.semibold).italic())
                                .foregroundStyle(.white.opacity(0.55))
                        )
                        .dynamicTypeSize(...DynamicTypeSize.accessibility2)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 120)
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.black.opacity(0.55))
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .strokeBorder(Color.cyan.opacity(0.35), lineWidth: 1)
                )
        )
        .accessibilityLabel("Signing transcript: \(session.captionDisplayText)")
    }

    private var conversationPanel: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Conversation")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.9))
                Spacer()
                if session.speech.isListening {
                    Label("Mic on", systemImage: "mic.fill")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.green)
                }
                Button("Clear") { session.clearHistory() }
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.7))
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(Color(white: 0.12))

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 10) {
                        ForEach(session.history) { turn in
                            turnBubble(turn)
                                .id(turn.id)
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
                    .padding(.vertical, 10)
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
            .background(Color(white: 0.08))
        }
    }

    private func turnBubble(_ turn: ConversationTurn) -> some View {
        HStack {
            if turn.role == .speaking { Spacer(minLength: 40) }
            VStack(alignment: turn.role == .speaking ? .trailing : .leading, spacing: 4) {
                Text(turn.role == .signing ? "Signing" : (turn.role == .speaking ? "You said" : "System"))
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(turn.role == .signing ? Color.cyan : Color.green)
                Text(turn.text)
                    .font(.body.weight(.medium))
                    .dynamicTypeSize(...DynamicTypeSize.accessibility2)
                    .foregroundStyle(.white)
                    .opacity(turn.isPartial ? 0.7 : 1)
                if !turn.gloss.isEmpty && turn.role == .signing {
                    Text(turn.gloss.joined(separator: " "))
                        .font(.caption2.monospaced())
                        .foregroundStyle(.white.opacity(0.5))
                }
            }
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(turn.role == .signing ? Color.cyan.opacity(0.18) : Color.green.opacity(0.18))
            )
            if turn.role == .signing { Spacer(minLength: 40) }
        }
    }
}
