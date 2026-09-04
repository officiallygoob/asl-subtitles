import SwiftUI

struct SettingsSheet: View {
    @ObservedObject var session: ASLSessionController
    @Environment(\.dismiss) private var dismiss
    @State private var serverURL: String = ""
    @State private var preferServer = true
    @State private var recordLabel: String = "HELLO"

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Toggle("Use recognition server", isOn: $preferServer)
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
                    Text("Continuous recognition")
                } footer: {
                    Text("Default: ws://127.0.0.1:8765/v1/stream — on a real device use your Mac’s LAN IP (Bonjour-friendly). Only landmark geometry is sent, never video.")
                }

                Section("Privacy") {
                    Label("Video stays on device", systemImage: "lock.shield.fill")
                    Text("The camera never uploads pixels. When a server is connected, only skeletal landmarks (hands/body/face points) are streamed. Offline mode uses on-device heuristics.")
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

                Section("Recognition") {
                    Toggle("Show landmark overlay", isOn: $session.showLandmarkDebug)
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
                    Button("Export all clips (JSONL)") {
                        _ = session.recorder.exportJSONL()
                    }
                    if let url = session.recorder.lastExportURL {
                        Text(url.lastPathComponent)
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                    }
                } header: {
                    Text("Landmark training")
                } footer: {
                    Text("Record labeled landmark sequences for Create ML Hand Action / fine-tunes. Highest-leverage path to friend-adapted recognition. Clips stay on device in Documents/LandmarkRecordings.")
                }

                Section("Honesty") {
                    Text("This app streams landmarks like DeepMind SL2T / MediaPipe Holistic architectures. It does **not** include Google’s proprietary SL2T model. Server accuracy depends on the weights you load (see MODELS.md). Offline mode is a small heuristic vocabulary only.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                if let error = session.lastErrorMessage {
                    Section("Diagnostics") {
                        Text(error).foregroundStyle(.red)
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .onAppear {
                serverURL = session.recognition.serverURLString
                preferServer = session.recognition.preferServer
            }
        }
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
