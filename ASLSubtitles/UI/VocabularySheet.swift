import SwiftUI

struct VocabularySheet: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Text("Starter MVP vocabulary — heuristic recognition from hand landmarks. Not fluent ASL. Results work best with clear lighting and the signing hand filling much of the frame.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                ForEach(VocabularyCatalog.grouped, id: \.0) { category, entries in
                    Section(category.rawValue) {
                        ForEach(entries) { entry in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(entry.word)
                                    .font(.headline)
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
