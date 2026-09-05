import Foundation

/// Client-side username format + abuse/derogatory guardrails for Call identity.
/// Server must re-validate when real accounts exist.
enum UsernameValidation: Equatable {
    case ok(String)
    /// User-facing reason shown under the username field.
    case rejected(String)
}

enum UsernameModerator {
    private static let minLength = 3
    private static let maxLength = 24

    /// Reserved product / privilege names (normalized match).
    private static let reserved: Set<String> = [
        "admin", "administrator", "admins", "mod", "moderator", "mods",
        "support", "helpsupport", "customersupport", "staff", "official",
        "aslsubtitles", "aslsubtitle", "asl", "system", "root", "owner",
        "null", "undefined", "everyone", "here", "channel", "server",
        "bot", "aslbot", "api", "security", "safety"
    ]

    /// Whole-token blocklist (normalized username equals term).
    private static let blockedTokens: Set<String> = [
        // Profanity / general insults
        "fuck", "fucker", "fucking", "motherfucker", "mf", "mofo",
        "shit", "bullshit", "asshole", "asshat", "dumbass", "jackass",
        "bastard", "bitch", "bitches", "sonofabitch",
        "damn", "dammit", "crap", "piss", "pissed",
        "dick", "dickhead", "cock", "cocksucker", "prick",
        "pussy", "twat", "cunt",
        "whore", "slut", "skank", "hoe", "thot",
        "retard", "retarded", "idiot", "moron", "stupid",
        "fag", "faggot", "dyke", "tranny", "shemale",
        "pedo", "pedophile", "rapist", "rape", "raper",
        "nazi", "hitler", "kkk",
        "killmyself", "kys", "suicide",
        "porn", "porno", "xxx", "onlyfans",
        "cum", "jizz", "semen", "orgasm", "anal", "oral",
        "blowjob", "handjob", "fellatio", "cunnilingus",
        "dildo", "vibrator", "hentai", "nsfw",
        // Hate / slurs (also covered as severe substrings where listed below)
        "nigger", "nigga", "negro", "chink", "gook", "spic", "wetback",
        "kike", "heeb", "paki", "raghead", "towelhead",
        "cracker", "honky", "gringo",
        "beaner", "coon", "darkie", "jigaboo", "porchmonkey",
        "redskin", "squaw", "injun",
        "gypsy", "gyppo",
        "tranny", "ladyboy",
        "abeed", "kaffir", "kafir",
        // Sexual / harassment
        "sex", "sexy", "nude", "nudes", "naked", "boobs", "tits", "titty",
        "penis", "vagina", "clit", "clitoris", "ballsack", "testicle",
        "masturbate", "wank", "wanker", "jerkoff",
        "incest", "zoophile", "bestiality",
        // Violence / threats flavor
        "terrorist", "isis", "alqaeda",
        "murder", "killer", "schoolshooter"
    ]

    /// Severe terms: reject if they appear as a substring of the normalized name.
    private static let severeSubstrings: Set<String> = [
        "nigger", "nigga", "chink", "gook", "spic", "wetback",
        "kike", "paki", "raghead", "towelhead",
        "faggot", "tranny", "shemale",
        "retard",
        "cunt", "motherfucker", "cocksucker",
        "pedophile", "pedo",
        "rape", "rapist",
        "hitler", "nazi",
        "killmyself",
        "porchmonkey", "jigaboo", "beaner",
        "kaffir", "abeed"
    ]

    /// Sanitize charset only (letters, digits, underscore); cap length.
    static func sanitizeFormat(_ raw: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_"))
        let filtered = String(raw.unicodeScalars.filter { allowed.contains($0) })
        return String(filtered.prefix(maxLength))
    }

    static func validate(_ raw: String) -> UsernameValidation {
        let cleaned = sanitizeFormat(raw.trimmingCharacters(in: .whitespacesAndNewlines))

        if cleaned.isEmpty {
            return .rejected("Choose a username (3–24 letters, numbers, or _).")
        }

        guard cleaned.count >= minLength else {
            return .rejected("Username must be at least \(minLength) characters.")
        }

        guard cleaned.count <= maxLength else {
            return .rejected("Username must be at most \(maxLength) characters.")
        }

        guard let first = cleaned.first, first.isLetter else {
            return .rejected("Username must start with a letter.")
        }

        // Phone-like: mostly digits (identity is username, not phone).
        let digits = cleaned.filter(\.isNumber).count
        let letters = cleaned.filter(\.isLetter).count
        if digits >= 7 || (digits > 0 && digits > letters) {
            return .rejected("Usernames can’t look like phone numbers.")
        }

        let forms = normalizedForms(cleaned)
        for form in forms {
            if reserved.contains(form) {
                return .rejected("That name is reserved.")
            }
            if blockedTokens.contains(form) {
                return .rejected("That username isn’t allowed.")
            }
            for severe in severeSubstrings where form.contains(severe) {
                return .rejected("That username isn’t allowed.")
            }
        }

        return .ok(cleaned)
    }

    /// Lowercase, strip `_`, map leetspeak. `1` expands to both `i` and `l`.
    private static func normalizedForms(_ cleaned: String) -> [String] {
        let lower = cleaned.lowercased().replacingOccurrences(of: "_", with: "")
        let mappedI = mapLeet(lower, oneAs: "i")
        let mappedL = mapLeet(lower, oneAs: "l")
        if mappedI == mappedL { return [mappedI] }
        return [mappedI, mappedL]
    }

    private static func mapLeet(_ s: String, oneAs: Character) -> String {
        String(s.map { ch -> Character in
            switch ch {
            case "0": return "o"
            case "1": return oneAs
            case "3": return "e"
            case "4": return "a"
            case "5": return "s"
            case "7": return "t"
            case "@": return "a"
            case "$": return "s"
            default: return ch
            }
        })
    }
}
