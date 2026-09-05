import SwiftUI

struct VocabularySheet: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Text("Offline heuristics cover a conversational subset (~\(VocabularyCatalog.heuristicCount) entries including letters). The continuous PoseLSTM targets a larger gloss set. Friend-specific LandmarkRecorder remains the best path for their dialect. Signs marked “Needs ML” are listed for the server / training — not reliable offline.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                ForEach(VocabularyCatalog.grouped, id: \.0) { category, entries in
                    Section(category.rawValue) {
                        ForEach(entries) { entry in
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Text(entry.word)
                                        .font(.headline)
                                    if !entry.heuristicSupported {
                                        Text("Needs ML")
                                            .font(.caption2.weight(.semibold))
                                            .padding(.horizontal, 6)
                                            .padding(.vertical, 2)
                                            .background(Color.orange.opacity(0.25), in: Capsule())
                                            .foregroundStyle(.orange)
                                    }
                                }
                                Text(entry.tip)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.vertical, 2)
                        }
                    }
                }
            }
            .navigationTitle("Supported Signs")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
