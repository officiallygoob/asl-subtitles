import SwiftUI

/// In-app video call tab — primary remote path (WebRTC/LiveKit to be wired).
/// FaceTime capture remains a secondary advanced option.
struct CallTabView: View {
    @ObservedObject var session: ASLSessionController
    @StateObject private var call = InAppCallController()
    @State private var showFaceTimeAdvanced = false
    @State private var showCopied = false

    var body: some View {
        NavigationStack {
            ZStack {
                AppTheme.canvas.ignoresSafeArea()
                VStack(spacing: 0) {
                    videoStage
                    captionChrome
                    if call.phase == .idle || call.phase == .ended {
                        lobby
                    } else {
                        inCallControls
                    }
                }
            }
            .navigationTitle("Call")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showFaceTimeAdvanced = true
                    } label: {
                        Image(systemName: "ellipsis.circle")
                            .foregroundStyle(AppTheme.captionSecondary)
                    }
                    .accessibilityLabel("Advanced: FaceTime capture")
                }
            }
            .sheet(isPresented: $showFaceTimeAdvanced) {
                NavigationStack {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("FaceTime capture is a workaround when you can’t use in-app calling. Prefer the Call tab’s private video link when possible.")
                                .font(.subheadline)
                                .foregroundStyle(AppTheme.captionSecondary)
                            CallModePanel(
                                call: session.callMode,
                                sharePlay: session.sharePlay,
                                captionPreview: session.captionDisplayText,
                                onStart: { session.beginCallMode() },
                                onStop: { session.endCallMode() },
                                onPiP: { session.callMode.startPiP() }
                            )
                        }
                        .padding()
                    }
                    .background(AppTheme.panel.ignoresSafeArea())
                    .navigationTitle("FaceTime (advanced)")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .confirmationAction) {
                            Button("Done") { showFaceTimeAdvanced = false }
                                .foregroundStyle(AppTheme.accent)
                        }
                    }
                }
                .preferredColorScheme(.dark)
                .presentationDetents([.medium, .large])
            }
        }
    }

    // MARK: - Video stage (local + remote placeholders)

    private var videoStage: some View {
        GeometryReader { geo in
            let remoteH = geo.size.height * 0.72
            ZStack(alignment: .bottomTrailing) {
                // Remote (friend) — recognition target once WebRTC frames arrive
                ZStack {
                    RoundedRectangle(cornerRadius: 28, style: .continuous)
                        .fill(
                            LinearGradient(
                                colors: [Color(white: 0.12), Color(white: 0.06)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                    VStack(spacing: 10) {
                        Image(systemName: call.phase == .connected ? "person.wave.2.fill" : "person.fill")
                            .font(.system(size: 44))
                            .foregroundStyle(AppTheme.accent.opacity(0.8))
                        Text(call.phase == .connected ? call.remoteName : "Waiting for friend")
                            .font(.headline)
                            .foregroundStyle(.white)
                        Text(call.statusMessage)
                            .font(.caption)
                            .foregroundStyle(AppTheme.captionMuted)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 24)
                    }
                }
                .frame(height: remoteH)
                .overlay {
                    RoundedRectangle(cornerRadius: 28, style: .continuous)
                        .strokeBorder(AppTheme.accent.opacity(0.25), lineWidth: 1)
                }
                .padding(.horizontal, 12)
                .padding(.top, 8)

                // Local PiP preview — reuse device camera when in-call
                ZStack {
                    if call.phase == .connected || call.phase == .waitingForPeer || call.phase == .connecting {
                        if call.isCameraOn {
                            CameraPreviewView(session: session.captureSession)
                                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                        } else {
                            RoundedRectangle(cornerRadius: 16, style: .continuous)
                                .fill(Color(white: 0.15))
                            Image(systemName: "video.slash.fill")
                                .foregroundStyle(.white.opacity(0.7))
                        }
                    } else {
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .fill(Color(white: 0.14))
                        Image(systemName: "person.crop.square")
                            .foregroundStyle(AppTheme.captionMuted)
                    }
                }
                .frame(width: 112, height: 152)
                .overlay {
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .strokeBorder(.white.opacity(0.25), lineWidth: 1)
                }
                .shadow(radius: 10, y: 4)
                .padding(.trailing, 22)
                .padding(.bottom, 18)
            }
        }
        .frame(height: 360)
    }

    private var captionChrome: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "captions.bubble.fill")
                    .foregroundStyle(AppTheme.accent)
                Text("Signing on this call")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(AppTheme.accent)
                    .textCase(.uppercase)
                Spacer()
                if session.callMode.isCallModeActive {
                    Text("FaceTime capture")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                }
            }
            let text = session.captionDisplayText
            Text(text.isEmpty ? "Captions appear when your friend signs…" : text)
                .font(.title2.weight(.bold))
                .foregroundStyle(text.isEmpty ? AppTheme.captionMuted : AppTheme.captionPrimary)
                .frame(maxWidth: AppTheme.captionMaxWidth, alignment: .leading)
                .minimumScaleFactor(0.7)
                .animation(.spring(response: 0.35, dampingFraction: 0.85), value: text)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .liquidGlassCard(cornerRadius: 20)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    private var lobby: some View {
        VStack(spacing: 14) {
            Text("In-app video calling")
                .font(.title3.weight(.bold))
                .foregroundStyle(.white)
            Text("Invite with a link or code and a username — we never ask for phone numbers. Signaling is stubbed; next step is WebRTC or LiveKit.")
                .font(.footnote)
                .foregroundStyle(AppTheme.captionMuted)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 8)

            HStack {
                Image(systemName: "person.crop.circle")
                    .foregroundStyle(AppTheme.accent)
                TextField("Username (no phone #)", text: $call.username)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
            }
            .padding(12)
            .background(AppTheme.panelElevated, in: Capsule())

            Button {
                UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                if session.permissionStatus == .authorized {
                    session.start()
                } else {
                    session.requestPermissionAndStart()
                }
                call.createInvite()
            } label: {
                Label("Share call link", systemImage: "square.and.arrow.up")
                    .font(.headline.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.black)
            .background(AppTheme.accent, in: Capsule())

            HStack {
                TextField("Join with link or code", text: $call.joinField)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .padding(12)
                    .background(AppTheme.panelElevated, in: Capsule())
                Button("Join") {
                    call.joinWithLinkOrCode()
                    session.start()
                }
                .font(.subheadline.weight(.bold))
                .foregroundStyle(AppTheme.accent)
                .disabled(call.joinField.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }

            Text("No Contacts access. No SMS. Just usernames + links.")
                .font(.caption2)
                .foregroundStyle(AppTheme.captionMuted)

            Spacer(minLength: 8)
        }
        .padding(16)
    }

    private var inCallControls: some View {
        VStack(spacing: 12) {
            if !call.inviteLink.isEmpty, call.phase == .waitingForPeer || call.phase == .connecting {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Code \(call.inviteCode)")
                            .font(.subheadline.weight(.bold).monospaced())
                            .foregroundStyle(AppTheme.accent)
                        Spacer()
                        Text("as \(call.displayName)")
                            .font(.caption)
                            .foregroundStyle(AppTheme.captionMuted)
                    }
                    HStack {
                        Text(call.inviteLink)
                            .font(.caption2.monospaced())
                            .foregroundStyle(AppTheme.captionSecondary)
                            .lineLimit(1)
                        Spacer()
                        Button(showCopied ? "Copied" : "Copy link") {
                            UIPasteboard.general.string = call.inviteLink
                            showCopied = true
                            UIImpactFeedbackGenerator(style: .light).impactOccurred()
                        }
                        .font(.caption.weight(.bold))
                        .foregroundStyle(AppTheme.accent)
                    }
                }
                .padding(12)
                .liquidGlassCard(cornerRadius: 14)
            }

            GlassChrome {
                HStack(spacing: 18) {
                    callButton(call.isMicOn ? "mic.fill" : "mic.slash.fill", active: call.isMicOn) {
                        call.toggleMic()
                        if call.isMicOn { session.setSpeechEnabled(true) } else { session.setSpeechEnabled(false) }
                    }
                    callButton(call.isCameraOn ? "video.fill" : "video.slash.fill", active: call.isCameraOn) {
                        call.toggleCamera()
                    }
                    callButton("arrow.triangle.2.circlepath.camera", active: false) {
                        session.toggleCamera()
                    }
                    Button {
                        UIImpactFeedbackGenerator(style: .heavy).impactOccurred()
                        call.endCall()
                    } label: {
                        Image(systemName: "phone.down.fill")
                            .font(.title3.weight(.bold))
                            .foregroundStyle(.white)
                            .frame(width: 58, height: 58)
                            .background(AppTheme.danger, in: Circle())
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 22)
                .padding(.vertical, 12)
                .liquidGlass(.interactive, in: Capsule())
            }
            .padding(.bottom, 12)
        }
        .padding(.horizontal, 16)
    }

    private func callButton(_ icon: String, active: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.body.weight(.semibold))
                .foregroundStyle(active ? AppTheme.accent : .white)
                .frame(width: 46, height: 46)
        }
        .buttonStyle(.plain)
    }
}
