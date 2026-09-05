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
                    tip: "Static ASL fingerspelling handshape. Letters like J and Z need motion and are weaker offline."
                )
            )
        }

        // tip, heuristicSupported
        typealias Row = (String, VocabularyEntry.Category, String, Bool)

        let rows: [Row] = [
            // Greetings
            ("Hello", .greeting, "Open palm near head; a small wave helps.", true),
            ("Hi", .greeting, "Similar to Hello — open palm wave near head.", true),
            ("Bye", .greeting, "Open palm with a side-to-side wave.", true),
            ("Good morning", .greeting, "GOOD + morning motion — better with server / friend data.", false),
            ("Good night", .greeting, "Flat hand down from face — approximate offline; better with ML.", false),
            ("See you later", .social, "SEE + LATER — multi-sign; server continuous model.", false),
            ("Nice to meet you", .social, "Phrase; needs continuous / friend model.", false),

            // Courtesy
            ("Thanks", .courtesy, "Flat hand from chin moving forward/down.", true),
            ("Please", .courtesy, "Open palm on chest with a small circle/motion.", true),
            ("Sorry", .courtesy, "Fist circling on the chest.", true),
            ("Excuse", .courtesy, "Flat hand brushing across palm — approximate.", true),
            ("Welcome", .courtesy, "Needs ML for reliable recognition.", false),

            // People / pronouns
            ("Me", .people, "Index finger pointing toward yourself.", true),
            ("You", .people, "Index finger pointing toward the other person.", true),
            ("We", .people, "Index sweeping inclusive arc (approx. lateral).", true),
            ("They", .people, "Index pointing with lateral sweep.", true),
            ("My", .people, "Flat hand on chest.", true),
            ("Your", .people, "Flat hand pushing toward other person.", true),
            ("Name", .people, "Two fingers tapping (H/U) with short motion.", true),
            ("Friend", .people, "Two index fingers hooking together.", true),
            ("Family", .people, "F handshapes circling (needs ML for accuracy).", false),
            ("Mother", .people, "Needs ML / friend data.", false),
            ("Father", .people, "Needs ML / friend data.", false),
            ("Sister", .people, "Needs ML / friend data.", false),
            ("Brother", .people, "Needs ML / friend data.", false),
            ("Baby", .people, "Needs ML / friend data.", false),
            ("Person", .people, "Needs ML / friend data.", false),

            // Questions
            ("What", .questions, "Open hands with a questioning side motion.", true),
            ("Where", .questions, "Index finger wagging side to side.", true),
            ("When", .questions, "Index circling on opposite hand — approximate.", true),
            ("Who", .questions, "Index circling near chin/mouth — approximate.", true),
            ("Why", .questions, "Y-hand near temple with small motion — approximate.", true),
            ("How", .questions, "Two fists together near center.", true),
            ("Which", .questions, "A-hands alternating up/down — approximate.", true),
            ("How much", .questions, "Needs ML.", false),
            ("How are you", .social, "HOW + YOU — continuous / phrase.", false),

            // Answers / stance
            ("Yes", .answers, "Fist nodding up and down.", true),
            ("No", .answers, "Thumb with index/middle flicking sideways.", true),
            ("OK", .answers, "Thumb and index forming a circle (F handshape).", true),
            ("Maybe", .answers, "Open palms rocking — approximate.", true),
            ("True", .answers, "Index moving forward from chin — approximate.", true),
            ("False", .answers, "Index brushing sideways under nose — approximate.", true),
            ("Right", .answers, "Needs ML.", false),
            ("Wrong", .answers, "Needs ML.", false),

            // Evaluative / everyday
            ("Good", .everyday, "Flat hand from chin moving down.", true),
            ("Bad", .everyday, "Flat hand from mouth turning out/down.", true),
            ("Fine", .everyday, "Open hand tips on chest — approximate.", true),
            ("Great", .everyday, "Open hand rising — approximate.", true),
            ("More", .everyday, "Fingertips tapping together.", true),
            ("Less", .everyday, "Flat hands lowering — approximate.", true),
            ("Same", .everyday, "Two indexes together sideways — approximate.", true),
            ("Different", .everyday, "Two indexes crossing apart — approximate.", true),
            ("Big", .everyday, "Needs ML.", false),
            ("Small", .everyday, "Needs ML.", false),
            ("New", .everyday, "Needs ML.", false),
            ("Old", .everyday, "Needs ML.", false),

            // Verbs
            ("Want", .verbs, "Claw hands pulling toward body — approximate.", true),
            ("Need", .verbs, "X-hand nodding down — approximate.", true),
            ("Help", .verbs, "Fist on open palm, lifting (two hands).", true),
            ("Understand", .verbs, "Index flicking up at forehead — approximate.", true),
            ("Know", .verbs, "Flat hand tips at forehead — approximate.", true),
            ("Don't know", .verbs, "Open hands flipping from forehead — approximate.", true),
            ("Like", .verbs, "Middle finger + thumb from chest — approximate.", true),
            ("Love", .verbs, "ILY handshape or crossed fists.", true),
            ("Go", .verbs, "Index pointing outward with motion.", true),
            ("Come", .verbs, "Index beckoning toward self.", true),
            ("Stop", .verbs, "Open palm held toward camera.", true),
            ("Wait", .verbs, "Open wiggling fingers — approximate.", true),
            ("Again", .verbs, "Bent hand tapping palm — approximate.", true),
            ("Slow", .verbs, "Hand sliding slowly over other — approximate.", true),
            ("Fast", .verbs, "Thumbs flicking quickly — approximate.", true),
            ("Look", .verbs, "V-hand from eyes outward — approximate.", true),
            ("Give", .verbs, "Needs ML.", false),
            ("Take", .verbs, "Needs ML.", false),
            ("Have", .verbs, "Needs ML.", false),
            ("Make", .verbs, "Needs ML.", false),
            ("Think", .verbs, "Needs ML.", false),
            ("Feel", .verbs, "Needs ML.", false),
            ("Remember", .verbs, "Needs ML.", false),
            ("Forget", .verbs, "Needs ML.", false),
            ("Tell", .verbs, "Needs ML.", false),
            ("Ask", .verbs, "Needs ML.", false),
            ("Call", .verbs, "Needs ML.", false),
            ("Meet", .verbs, "Needs ML.", false),
            ("Leave", .verbs, "Needs ML.", false),
            ("Stay", .verbs, "Needs ML.", false),
            ("Work", .verbs, "S-hands tapping at wrists — approximate.", true),
            ("Play", .verbs, "Needs ML.", false),
            ("Read", .verbs, "Needs ML.", false),
            ("Write", .verbs, "Pinch hand writing motion — approximate.", true),
            ("Spell", .verbs, "Fingerspelling motion / lateral point — approximate.", true),

            // Food & drink
            ("Eat", .food, "Pinch hand toward mouth.", true),
            ("Drink", .food, "C-hand tilting toward mouth.", true),
            ("Hungry", .food, "Claw down chest — approximate.", true),
            ("Thirsty", .food, "Needs ML.", false),
            ("Water", .food, "Needs ML.", false),
            ("Coffee", .food, "Needs ML.", false),
            ("Food", .food, "Needs ML.", false),
            ("Breakfast", .food, "Needs ML.", false),
            ("Lunch", .food, "Needs ML.", false),
            ("Dinner", .food, "Needs ML.", false),

            // Feelings
            ("Happy", .feelings, "Flat hands brushing up chest — approximate.", true),
            ("Sad", .feelings, "Open hands down face — approximate.", true),
            ("Tired", .feelings, "Fingertips at shoulders dropping — approximate.", true),
            ("Hot", .feelings, "Claw from mouth outward — approximate.", true),
            ("Cold", .feelings, "Fists shaking near shoulders — approximate.", true),
            ("Angry", .feelings, "Needs ML.", false),
            ("Scared", .feelings, "Needs ML.", false),
            ("Excited", .feelings, "Needs ML.", false),
            ("Sick", .feelings, "Needs ML.", false),
            ("Hurt", .feelings, "Needs ML.", false),
            ("Better", .feelings, "Needs ML.", false),
            ("Worse", .feelings, "Needs ML.", false),

            // Time
            ("Time", .time, "Index tapping wrist — approximate.", true),
            ("Today", .time, "Y-hands dropping — approximate.", true),
            ("Tomorrow", .time, "A-hand forward from cheek — approximate.", true),
            ("Yesterday", .time, "Needs ML.", false),
            ("Now", .time, "Needs ML.", false),
            ("Later", .time, "L-hand twisting — approximate via circle.", true),
            ("Morning", .time, "Needs ML.", false),
            ("Night", .time, "Needs ML.", false),
            ("Week", .time, "Needs ML.", false),
            ("Month", .time, "Needs ML.", false),
            ("Year", .time, "Needs ML.", false),
            ("Hour", .time, "Needs ML.", false),
            ("Minute", .time, "Needs ML.", false),

            // Places
            ("Home", .places, "Flat fingertips to cheek/mouth area — approximate.", true),
            ("School", .places, "Clapping flat hands — approximate.", true),
            ("Work", .places, "See verbs — S-hands tapping.", true),
            ("Store", .places, "Needs ML.", false),
            ("Hospital", .places, "Needs ML.", false),
            ("Bathroom", .places, "Needs ML.", false),
            ("Outside", .places, "Needs ML.", false),
            ("Inside", .places, "Needs ML.", false),
            ("Here", .places, "Needs ML.", false),
            ("There", .places, "Needs ML.", false),
            ("City", .places, "Needs ML.", false),

            // Numbers (finger counts — offline heuristics)
            ("One", .numbers, "Index only.", true),
            ("Two", .numbers, "Index + middle.", true),
            ("Three", .numbers, "Thumb + index + middle.", true),
            ("Four", .numbers, "Four fingers, no thumb.", true),
            ("Five", .numbers, "Open hand all fingers.", true),
            ("Six", .numbers, "Needs careful handshape; weaker offline.", false),
            ("Seven", .numbers, "Needs ML.", false),
            ("Eight", .numbers, "Needs ML.", false),
            ("Nine", .numbers, "Needs ML.", false),
            ("Ten", .numbers, "Thumb shake — approximate.", true),

            // Social / repair
            ("Again", .social, "See verbs.", true),
            ("Slow", .social, "Ask signer to slow — see verbs.", true),
            ("Understand?", .social, "UNDERSTAND as question — continuous helps.", false),
            ("Don't understand", .social, "Needs ML / phrase.", false),
            ("Repeat", .social, "Same idea as Again.", false),
            ("What did you say", .social, "Phrase — needs continuous model.", false),
        ]

        for (word, cat, tip, heur) in rows {
            items.append(VocabularyEntry(word: word, category: cat, tip: tip, heuristicSupported: heur))
        }

        // Explicit “needs ML” mirror for glosses the server lists but heuristics can’t do well.
        let mlOnly = [
            "DEAF", "HEARING", "SIGN", "ASL", "ENGLISH", "PHONE", "TEXT", "EMAIL",
            "CAR", "BUS", "TRAIN", "WALK", "RUN", "SIT", "STAND", "SLEEP", "WAKE",
            "BOOK", "MONEY", "BUY", "SELL", "PAY", "FREE", "BUSY", "READY",
            "IMPORTANT", "PROBLEM", "QUESTION", "ANSWER", "IDEA", "COLOR",
            "RED", "BLUE", "GREEN", "BLACK", "WHITE", "RAIN", "SUN", "WEATHER",
            "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"
        ]
        for word in mlOnly {
            items.append(
                VocabularyEntry(
                    word: word.capitalized,
                    category: .needsML,
                    tip: "Listed for continuous PoseLSTM / friend LandmarkRecorder — not offline heuristics.",
                    heuristicSupported: false
                )
            )
        }

        return items
    }()

    static var grouped: [(VocabularyEntry.Category, [VocabularyEntry])] {
        VocabularyEntry.Category.allCases.compactMap { cat in
            let list = entries.filter { $0.category == cat }
            return list.isEmpty ? nil : (cat, list)
        }
    }

    static var heuristicCount: Int {
        entries.filter(\.heuristicSupported).count
    }

    static var totalGlossLikeCount: Int {
        entries.filter { $0.category != .fingerspelling }.count
    }
}
