import SwiftUI

struct SettingsSheet: View {
    @ObservedObject var session: ASLSessionController
    @Environment(\.dismiss) private var dismiss
    @State private var serverURL: String = ""
    @State private var preferServer = false
    @State private var recordLabel: String = "HELLO"

    var body: some View {
        NavigationStack {
            List {
                Section {
                    LabeledContent("On-device model") {
                        Text(session.recognition.onDeviceModelName)
                            .foregroundStyle(.secondary)
                            .font(.caption)
                    }
                    Toggle("Use LAN recognition server (dev)", isOn: $preferServer)
                        .onChange(of: preferServer) { _, value in
                            session.recognition.preferServer = value
                        }
                    TextField("WebSocket URL", text: $serverURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                        .font(.footnote.monospaced())
                    Button("Apply & reconnect") {
                        session.recognition.serverURLString = serverURL.trimmingCharacters(in: .whitespacesAndNewlines)
                    }
                    LabeledContent("Status") {
                        Text(statusLabel)
                            .foregroundStyle(.secondary)
                    }
                    if !session.recognition.modelName.isEmpty {
                        LabeledContent("Model") {
                            Text(session.recognition.modelName)
                                .foregroundStyle(.secondary)
                        }
                    }
                } header: {
                    Text("Recognition (on-device first)")
                } footer: {
                    Text("Privacy-first default: camera → Vision landmarks → on-device Core ML / heuristics → subtitles. Nothing leaves the phone. LAN WebSocket is optional for developers only.")
                }

                Section {
                    Text("Prefer the Call tab for private in-app video. FaceTime screen capture is a workaround only.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    Button("Open FaceTime capture coach") {
                        session.beginCallMode()
                        dismiss()
                    }
                } header: {
                    Text("FaceTime (advanced)")
                }

                Section {
                    Toggle("Show landmark overlay", isOn: $session.showLandmarkDebug)
                    Toggle("NMM badges (brow / shake / lean)", isOn: $session.showNMMBadges)
                        .disabled(!session.showLandmarkDebug)
                } header: {
                    Text("Debug")
                } footer: {
                    Text("Uses face + body cues (questions, negation, emphasis), not hands alone. Phone Vision landmarks are approximate vs studio MoCap.")
                }

                Section("Privacy") {
                    Label("Video stays on device", systemImage: "lock.shield.fill")
                        .foregroundStyle(AppTheme.accent)
                    Text("Default path never uploads. Camera → Vision landmarks → on-device Core ML → subtitles. LAN server (if enabled) streams landmarks only — never video pixels.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Speech → text") {
                    Toggle("Listen while in conversation", isOn: Binding(
                        get: { session.speechEnabled },
                        set: { session.setSpeechEnabled($0) }
                    ))
                    Text("Uses Apple’s Speech framework so you can speak and your deaf friend can read the transcript. Requires microphone permission.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section {
                    TextField("Gloss label (e.g. HELLO)", text: $recordLabel)
                        .textInputAutocapitalization(.characters)
                    if session.recorder.isRecording {
                        Text("Recording… \(session.recorder.frameCount) frames")
                            .foregroundStyle(.orange)
                        Button("Stop & save clip", role: .destructive) {
                            _ = session.recorder.stopAndSave()
                        }
                    } else {
                        Button("Start recording") {
                            session.recorder.start(label: recordLabel)
                        }
                        .disabled(recordLabel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                    if session.recorder.targetTakes > 0 {
                        Text("Takes for \(session.recorder.currentLabel.isEmpty ? recordLabel : session.recorder.currentLabel): \(session.recorder.takesForCurrentLabel)/\(session.recorder.targetTakes)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Button("Export all clips (JSONL)") {
                        _ = session.recorder.exportJSONL()
                    }
                    Button("Export Create ML Hand Action CSV") {
                        _ = session.recorder.exportCreateMLCSV()
                    }
                    if let url = session.recorder.lastExportURL {
                        Text(url.lastPathComponent)
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                    }
                } header: {
                    Text("Train / Capture (optional adaptation)")
                } footer: {
                    Text("Optional friend adaptation: pick a gloss, record several takes (hands+face+body+NMM), export JSON/JSONL. Clips stay on-device. Primary accuracy comes from the bundled Core ML model trained offline on public pose data.")
                }

                Section("Accuracy & privacy") {
                    Text("Privacy: captions run on-device (Vision → Core ML). No video/landmarks leave the phone unless you opt into the LAN server. Accuracy: bundled PoseLSTM was trained offline on public WLASL100 pose landmarks (+ synth fill). Friend recordings are optional adaptation. Phone NMMs are soft cues — not studio MoCap. Not Google SL2T; conversational ASL is still unsolved.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                if let error = session.lastErrorMessage {
                    Section("Diagnostics") {
                        Text(error).foregroundStyle(.red)
                    }
                }
            }
            .scrollContentBackground(.hidden)
            .background(AppTheme.panel.ignoresSafeArea())
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                        .fontWeight(.semibold)
                        .foregroundStyle(AppTheme.accent)
                }
            }
            .onAppear {
                serverURL = session.recognition.serverURLString
                preferServer = session.recognition.preferServer
            }
        }
        .preferredColorScheme(.dark)
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
    }

    private var statusLabel: String {
        switch session.recognition.state {
        case .connected: return "Connected"
        case .connecting: return "Connecting…"
        case .offlineFallback: return "Offline fallback"
        case .disconnected: return "Disconnected"
        case .error(let m): return m
        }
    }
}
