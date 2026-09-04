import SwiftUI

struct SettingsSheet: View {
    @ObservedObject var session: ASLSessionController
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section("Privacy") {
                    Label("All processing stays on this device", systemImage: "lock.shield.fill")
                    Text("Camera frames are analyzed with Apple Vision on-device. No video or landmarks are uploaded.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Recognition") {
                    Toggle("Show hand landmarks", isOn: $session.showLandmarkDebug)
                    Text("Debug overlay draws Vision joint points. Useful while teaching the camera your signing style.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("About this MVP") {
                    Text("ASL Subtitles is a limited starter: fingerspelling A–Z plus ~20 everyday signs via hand-pose heuristics. It is not a substitute for interpreters or fluent ASL understanding.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                if let error = session.lastErrorMessage {
                    Section("Camera") {
                        Text(error)
                            .foregroundStyle(.red)
                    }
                }
            }
            .navigationTitle("About")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
