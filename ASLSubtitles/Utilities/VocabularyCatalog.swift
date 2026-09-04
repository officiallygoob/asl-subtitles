import Foundation

enum VocabularyCatalog {
    static let entries: [VocabularyEntry] = {
        var items: [VocabularyEntry] = []

        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for ch in letters {
            items.append(
                VocabularyEntry(
                    word: String(ch),
                    category: .fingerspelling,
                    tip: "Static ASL fingerspelling handshape. Letters like J and Z need motion and are weaker in this MVP."
                )
            )
        }

        let everyday: [(String, VocabularyEntry.Category, String)] = [
            ("Hello", .greeting, "Open palm near head; a small wave helps."),
            ("Bye", .greeting, "Open palm with a side-to-side wave."),
            ("Thanks", .courtesy, "Flat hand from chin moving forward/down."),
            ("Please", .courtesy, "Open palm on chest with a small circle/motion."),
            ("Sorry", .courtesy, "Fist circling on the chest."),
            ("Yes", .everyday, "Fist nodding up and down."),
            ("No", .everyday, "Thumb with index/middle flicking sideways."),
            ("Help", .everyday, "Fist on open palm, lifting upward (two hands)."),
            ("Name", .everyday, "Two fingers tapping (H/U handshape) with short motion."),
            ("Friend", .people, "Two index fingers hooking together."),
            ("Love", .people, "ILY handshape (thumb+index+pinky) or crossed fists."),
            ("How", .everyday, "Two fists together near center."),
            ("You", .people, "Index finger pointing toward the other person."),
            ("Me", .people, "Index finger pointing toward yourself."),
            ("Good", .everyday, "Flat hand from chin moving down."),
            ("Bad", .everyday, "Flat hand from mouth turning out/down."),
            ("More", .everyday, "Fingertips tapping together."),
            ("What", .everyday, "Open hands with a questioning side motion."),
            ("Where", .everyday, "Index finger wagging side to side."),
            ("OK", .everyday, "Thumb and index forming a circle (F handshape)."),
            ("Stop", .everyday, "Open palm held toward camera.")
        ]

        for (word, cat, tip) in everyday {
            items.append(VocabularyEntry(word: word, category: cat, tip: tip))
        }

        return items
    }()

    static var grouped: [(VocabularyEntry.Category, [VocabularyEntry])] {
        VocabularyEntry.Category.allCases.compactMap { cat in
            let list = entries.filter { $0.category == cat }
            return list.isEmpty ? nil : (cat, list)
        }
    }
}
