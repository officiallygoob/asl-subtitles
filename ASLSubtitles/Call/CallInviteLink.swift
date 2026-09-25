import Foundation

/// Invite identity for a 1:1 call: a room code plus an optional inviter username.
/// Links look like `aslsubtitles://call/<roomID>?from=<username>`.
struct CallInvite: Equatable {
    var roomID: String
    var fromUsername: String?

    var link: String {
        CallInviteLink.make(roomID: roomID, from: fromUsername ?? "")
    }
}

enum CallInviteLink {
    static let scheme = "aslsubtitles"

    static func make(roomID: String, from username: String) -> String {
        var components = URLComponents()
        components.scheme = scheme
        components.host = "call"
        components.path = "/" + roomID
        if !username.isEmpty {
            components.queryItems = [URLQueryItem(name: "from", value: username)]
        }
        if let built = components.string {
            return built
        }
        if username.isEmpty {
            return "\(scheme)://call/\(roomID)"
        }
        let encoded = username.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? username
        return "\(scheme)://call/\(roomID)?from=\(encoded)"
    }

    static func parse(url: URL) -> CallInvite? {
        guard url.scheme?.lowercased() == scheme else { return nil }
        guard url.host?.lowercased() == "call" else { return nil }
        let rawRoom = url.path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard let roomID = normalizeRoomID(rawRoom) else { return nil }
        let from = URLComponents(url: url, resolvingAgainstBaseURL: false)?
            .queryItems?
            .first(where: { $0.name == "from" })?
            .value
        return CallInvite(roomID: roomID, fromUsername: acceptedUsername(from))
    }

    /// Accepts a bare code (`ab12cd34`) or a full `aslsubtitles://call/...` link.
    static func parse(text: String) -> CallInvite? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        if let url = URL(string: trimmed), url.scheme?.lowercased() == scheme {
            return parse(url: url)
        }

        let pieces = trimmed.split(separator: "?", maxSplits: 1, omittingEmptySubsequences: false)
        let beforeQuery = String(pieces.first ?? Substring(trimmed))
        let segment = beforeQuery.split(separator: "/").last.map(String.init) ?? beforeQuery
        guard let roomID = normalizeRoomID(segment) else { return nil }

        var fromRaw: String?
        if pieces.count == 2 {
            let query = pieces[1]
            for part in query.split(separator: "&") {
                let pair = part.split(separator: "=", maxSplits: 1)
                guard pair.first?.lowercased() == "from" else { continue }
                let value = pair.count > 1 ? String(pair[1]) : ""
                fromRaw = value.removingPercentEncoding ?? value
            }
        }
        return CallInvite(roomID: roomID, fromUsername: acceptedUsername(fromRaw))
    }

    /// Lowercase room codes, 4–64 letters or digits (the app mints 8 hex characters).
    static func normalizeRoomID(_ raw: String) -> String? {
        let lowered = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard (4...64).contains(lowered.count) else { return nil }
        let allowed = CharacterSet.alphanumerics
        guard lowered.unicodeScalars.allSatisfy({ allowed.contains($0) }) else { return nil }
        return lowered
    }

    /// Inviter hint only. Invalid names are dropped and never stored as our username.
    private static func acceptedUsername(_ raw: String?) -> String? {
        guard let raw else { return nil }
        if case .ok(let good) = UsernameModerator.validate(raw) {
            return good
        }
        return nil
    }
}
