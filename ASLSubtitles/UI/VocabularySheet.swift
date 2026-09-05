import SwiftUI

struct VocabularySheet: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        Label("Face + body cues", systemImage: "face.smiling.inverse")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(AppTheme.accent)
                        Text("Uses face + body cues (questions, negation, emphasis), not hands alone. Offline heuristics cover a conversational subset (~\(VocabularyCatalog.heuristicCount) entries including letters). Raised brows can turn a gloss into a question; head shake / frown can negate; lean adds emphasis.")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 4)
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
            .scrollContentBackground(.hidden)
            .background(AppTheme.panel.ignoresSafeArea())
            .navigationTitle("Supported Signs")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                        .fontWeight(.semibold)
                        .foregroundStyle(AppTheme.accent)
                }
            }
        }
        .preferredColorScheme(.dark)
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
    }
}
