import LiveKit
import SwiftUI

/// In-app 1:1 LiveKit call. FaceTime capture remains a secondary advanced option.
struct CallTabView: View {
    @ObservedObject var session: ASLSessionController
    @StateObject private var call = InAppCallController()
    @EnvironmentObject private var callLinks: CallDeepLinkCenter
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
            .onAppear {
                bindCallSession()
                consumePendingInvite()
            }
            .onChange(of: callLinks.pendingInvite) { _, _ in
                consumePendingInvite()
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
                // Remote (friend). Landmarks from this track go to HolisticPoseDetector on device.
                ZStack {
                    RoundedRectangle(cornerRadius: 28, style: .continuous)
                        .fill(
                            LinearGradient(
                                colors: [Color(white: 0.12), Color(white: 0.06)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                    if call.showsLocalPreview, let track = call.remoteVideoTrack, !call.remoteCameraMuted {
                        SwiftUIVideoView(track, layoutMode: .fill, mirrorMode: .off)
                            .clipShape(RoundedRectangle(cornerRadius: 28, style: .continuous))
                        VStack {
                            Spacer()
                            HStack {
                                Text(call.remoteName)
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.white)
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 6)
                                    .background(.black.opacity(0.45), in: Capsule())
                                Spacer()
                            }
                            .padding(16)
                        }
                    } else {
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
                }
                .frame(height: remoteH)
                .overlay {
                    RoundedRectangle(cornerRadius: 28, style: .continuous)
                        .strokeBorder(AppTheme.accent.opacity(0.25), lineWidth: 1)
                }
                .padding(.horizontal, 12)
                .padding(.top, 8)

                // Local PiP — LiveKit camera, not the in-person recognition session.
                ZStack {
                    if call.showsLocalPreview {
                        if call.isCameraOn, let track = call.localVideoTrack {
                            SwiftUIVideoView(
                                track,
                                layoutMode: .fill,
                                mirrorMode: call.prefersFrontCamera ? .mirror : .off
                            )
                            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                        } else if call.isCameraOn {
                            RoundedRectangle(cornerRadius: 16, style: .continuous)
                                .fill(Color(white: 0.15))
                            ProgressView()
                                .tint(.white)
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
            Text("Invite with a link or code and a username — we never ask for phone numbers. Video uses LiveKit; sign recognition stays on this phone.")
                .font(.footnote)
                .foregroundStyle(AppTheme.captionMuted)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 8)

            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Image(systemName: "person.crop.circle")
                        .foregroundStyle(AppTheme.accent)
                    TextField("Username (no phone #)", text: $call.username)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .textContentType(.username)
                        .keyboardType(.asciiCapable)
                }
                .padding(12)
                .background(AppTheme.panelElevated, in: Capsule())

                if let err = call.usernameError {
                    Text(err)
                        .font(.caption)
                        .foregroundStyle(AppTheme.danger)
                        .padding(.horizontal, 8)
                } else if !call.username.isEmpty && !call.hasValidUsername {
                    Text("3–24 chars, start with a letter, letters/numbers/_ only.")
                        .font(.caption)
                        .foregroundStyle(AppTheme.captionSecondary)
                        .padding(.horizontal, 8)
                }
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Token server (LAN address, not a phone number)")
                    .font(.caption2)
                    .foregroundStyle(AppTheme.captionMuted)
                    .padding(.horizontal, 8)
                TextField("http://192.168.1.10:8787/v1/token", text: $call.tokenServerURL)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .textContentType(.URL)
                    .keyboardType(.URL)
                    .padding(12)
                    .background(AppTheme.panelElevated, in: Capsule())
            }

            if let err = call.lastError {
                Text(err)
                    .font(.caption)
                    .foregroundStyle(AppTheme.danger)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 8)
            }

            Button {
                UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                call.createInvite()
            } label: {
                Label("Share call link", systemImage: "square.and.arrow.up")
                    .font(.headline.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.black)
            .background(call.hasValidUsername ? AppTheme.accent : AppTheme.accent.opacity(0.35), in: Capsule())
            .disabled(!call.hasValidUsername)

            HStack {
                TextField("Join with link or code", text: $call.joinField)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .padding(12)
                    .background(AppTheme.panelElevated, in: Capsule())
                Button("Join") {
                    call.joinWithLinkOrCode()
                }
                .font(.subheadline.weight(.bold))
                .foregroundStyle(AppTheme.accent)
                .disabled(!call.hasValidUsername || call.joinField.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
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
            if let err = call.lastError {
                Text(err)
                    .font(.caption2)
                    .foregroundStyle(AppTheme.danger)
                    .multilineTextAlignment(.center)
            }
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
                    }
                    callButton(call.isCameraOn ? "video.fill" : "video.slash.fill", active: call.isCameraOn) {
                        call.toggleCamera()
                    }
                    callButton("arrow.triangle.2.circlepath.camera", active: false) {
                        call.flipCamera()
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

    private func bindCallSession() {
        call.prepareForCallMedia = { await session.camera.stopAndWait() }
        call.releaseCallMedia = {
            if session.permissionStatus == .authorized {
                session.start()
            }
        }
        call.onRemoteLandmarks = { frame in
            session.noteRemoteCallLandmarks(frame)
        }
    }

    private func consumePendingInvite() {
        bindCallSession()
        guard let invite = callLinks.pendingInvite else { return }
        callLinks.pendingInvite = nil
        call.applyIncomingInvite(invite)
    }
}
