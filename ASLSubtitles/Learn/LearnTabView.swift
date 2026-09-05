import SwiftUI

/// Flashcard Learn ASL tab — prompt → sign → score with existing recognition.
struct LearnTabView: View {
    @ObservedObject var session: ASLSessionController
    @StateObject private var learn = LearnSessionController()
    @ObservedObject private var progress = LearnProgressStore.shared

    var body: some View {
        NavigationStack {
            ZStack {
                AppTheme.canvas.ignoresSafeArea()
                VStack(spacing: 0) {
                    cameraHeader
                    cardArea
                    controls
                }
            }
            .navigationTitle("Learn")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Text("\(progress.masteredCount) practiced")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(AppTheme.captionMuted)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Text(learn.progressLabel)
                        .font(.caption.monospaced())
                        .foregroundStyle(AppTheme.captionSecondary)
                }
            }
            .onAppear {
                session.prepare()
                if session.permissionStatus == .authorized {
                    session.start()
                }
                if learn.queue.isEmpty { learn.startDeck() }
                session.learnConsumer = { frame in
                    learn.consume(frame: frame)
                }
            }
            .onDisappear {
                session.learnConsumer = nil
            }
        }
    }

    private var cameraHeader: some View {
        ZStack {
            CameraPreviewView(session: session.captureSession)
                .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: 24, style: .continuous)
                        .strokeBorder(AppTheme.accent.opacity(learn.phase == .watching ? 0.7 : 0.2), lineWidth: learn.phase == .watching ? 2 : 1)
                }
                .shadow(color: AppTheme.accent.opacity(learn.phase == .watching ? 0.25 : 0), radius: 16)

            if session.showLandmarkDebug {
                HolisticLandmarkOverlay(
                    hands: session.latestHands,
                    bodyJoints: session.latestBodyJoints,
                    face: session.latestFaceJoints,
                    nmm: session.showNMMBadges ? session.latestNMM : nil
                )
                .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
                .allowsHitTesting(false)
            }

            VStack {
                Spacer()
                if learn.phase == .watching {
                    VStack(spacing: 6) {
                        ProgressView(value: learn.holdProgress)
                            .tint(AppTheme.accent)
                        Text(learn.liveGuess.isEmpty ? "Watching…" : "\(learn.liveGuess) · \(Int(learn.liveConfidence * 100))%")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.white)
                    }
                    .padding(10)
                    .liquidGlassCard(cornerRadius: 12)
                    .padding(12)
                }
            }
        }
        .frame(height: 220)
        .padding(.horizontal, 12)
        .padding(.top, 8)
    }

    private var cardArea: some View {
        Group {
            if let card = learn.current {
                VStack(spacing: 16) {
                    Image(systemName: card.symbolName)
                        .font(.system(size: 36))
                        .foregroundStyle(AppTheme.accent)
                        .padding(.top, 8)

                    Text(card.title)
                        .font(.system(size: 48, weight: .bold, design: .rounded))
                        .foregroundStyle(.white)
                        .minimumScaleFactor(0.5)
                        .lineLimit(2)
                        .multilineTextAlignment(.center)

                    if card.isLetter {
                        Text("Fingerspelling")
                            .font(.caption.weight(.bold))
                            .foregroundStyle(AppTheme.accent)
                            .textCase(.uppercase)
                    }

                    Text(card.hint)
                        .font(.subheadline)
                        .foregroundStyle(AppTheme.captionSecondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 12)

                    if !learn.feedback.isEmpty {
                        Text(learn.feedback)
                            .font(.headline.weight(.semibold))
                            .foregroundStyle(learn.phase == .correct ? AppTheme.speakingRole : AppTheme.captionPrimary)
                            .padding(.top, 4)
                    }

                    phaseBadge
                }
                .padding(20)
                .frame(maxWidth: .infinity)
                .liquidGlassCard(cornerRadius: 24)
                .padding(.horizontal, 12)
                .padding(.top, 12)
                .animation(.spring(response: 0.4, dampingFraction: 0.85), value: learn.phase)
                .animation(.spring(response: 0.4, dampingFraction: 0.85), value: card.id)
            } else {
                Text("No cards")
                    .foregroundStyle(AppTheme.captionMuted)
                    .padding()
            }
        }
    }

    @ViewBuilder
    private var phaseBadge: some View {
        switch learn.phase {
        case .prompt:
            Label("Study the sign, then get ready", systemImage: "eyes")
                .font(.caption.weight(.semibold))
                .foregroundStyle(AppTheme.captionMuted)
        case .watching:
            Label("Sign now — hold until we confirm", systemImage: "camera.viewfinder")
                .font(.caption.weight(.semibold))
                .foregroundStyle(AppTheme.accent)
        case .correct:
            Label("Correct", systemImage: "checkmark.circle.fill")
                .font(.caption.weight(.bold))
                .foregroundStyle(AppTheme.speakingRole)
        case .tryAgain:
            Label("Try again", systemImage: "arrow.counterclockwise")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.orange)
        }
    }

    private var controls: some View {
        VStack(spacing: 12) {
            GlassChrome {
                HStack(spacing: 12) {
                    Button {
                        learn.again()
                    } label: {
                        Text("Again")
                            .font(.subheadline.weight(.semibold))
                            .padding(.horizontal, 16)
                            .padding(.vertical, 12)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.orange)

                    if learn.phase == .prompt {
                        Button {
                            if session.permissionStatus != .authorized {
                                session.requestPermissionAndStart()
                            }
                            learn.imReady()
                        } label: {
                            Text("I'm ready")
                                .font(.headline.weight(.bold))
                                .foregroundStyle(.black)
                                .padding(.horizontal, 28)
                                .padding(.vertical, 14)
                                .background(AppTheme.accent, in: Capsule())
                        }
                        .buttonStyle(.plain)
                    } else if learn.phase == .watching {
                        Button {
                            learn.skip()
                        } label: {
                            Text("Skip")
                                .font(.subheadline.weight(.semibold))
                                .padding(.horizontal, 16)
                                .padding(.vertical, 12)
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(AppTheme.captionSecondary)
                    } else {
                        Button {
                            learn.skip()
                        } label: {
                            Text("Next")
                                .font(.headline.weight(.bold))
                                .foregroundStyle(.black)
                                .padding(.horizontal, 28)
                                .padding(.vertical, 14)
                                .background(AppTheme.accent, in: Capsule())
                        }
                        .buttonStyle(.plain)
                    }

                    Button {
                        learn.gotItManual()
                    } label: {
                        Text("Got it")
                            .font(.subheadline.weight(.semibold))
                            .padding(.horizontal, 16)
                            .padding(.vertical, 12)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(AppTheme.speakingRole)
                }
                .padding(10)
                .liquidGlass(.interactive, in: Capsule())
            }

            Button("Shuffle deck") {
                learn.startDeck()
            }
            .font(.caption.weight(.semibold))
            .foregroundStyle(AppTheme.captionMuted)
            .padding(.bottom, 10)
        }
        .padding(.horizontal, 12)
        .padding(.top, 10)
    }
}
