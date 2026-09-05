import SwiftUI

struct HistorySheet: View {
    @ObservedObject var session: ASLSessionController
    @ObservedObject private var store = ConversationStore.shared
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Text("Past transcripts stay on this iPhone only. Spotlight and Siri can open them via App Intents when indexed. Clear anytime.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                if store.conversations.isEmpty {
                    Section {
                        Text("No saved conversations yet.")
                            .foregroundStyle(.secondary)
                    }
                } else {
                    Section("On this device") {
                        ForEach(store.conversations) { item in
                            Button {
                                session.openPersistedConversation(id: item.id)
                                dismiss()
                            } label: {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(item.title)
                                        .font(.headline)
                                        .foregroundStyle(.white)
                                    Text(item.preview)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(2)
                                    Text(item.updatedAt.formatted(date: .abbreviated, time: .shortened))
                                        .font(.caption2)
                                        .foregroundStyle(.tertiary)
                                }
                                .padding(.vertical, 2)
                            }
                        }
                        .onDelete { indexSet in
                            for i in indexSet {
                                store.delete(id: store.conversations[i].id)
                            }
                        }
                    }
                }
            }
            .scrollContentBackground(.hidden)
            .background(AppTheme.panel.ignoresSafeArea())
            .navigationTitle("Past Conversations")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("New") {
                        session.startNewConversation()
                        dismiss()
                    }
                    .foregroundStyle(AppTheme.accent)
                }
                ToolbarItem(placement: .bottomBar) {
                    Button("Clear All", role: .destructive) {
                        store.clearAll()
                    }
                    .disabled(store.conversations.isEmpty)
                }
            }
        }
        .preferredColorScheme(.dark)
        .presentationDetents([.medium, .large])
    }
}
