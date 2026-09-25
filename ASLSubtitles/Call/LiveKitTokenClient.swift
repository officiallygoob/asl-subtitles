import Foundation

struct LiveKitConnectionDetails: Decodable, Equatable {
    var url: String
    var token: String
    var room: String
    var identity: String
}

enum LiveKitTokenError: LocalizedError {
    case invalidURL
    case server(String)
    case badResponse

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Enter a token server URL, like http://192.168.1.10:8787/v1/token."
        case .server(let message):
            return message
        case .badResponse:
            return "The token server returned something unexpected."
        }
    }
}

/// Asks the LAN token server for a room-scoped LiveKit JWT.
/// The request body is a username and a room code — never a phone number.
enum LiveKitTokenClient {
    struct ErrorBody: Decodable {
        var error: String
    }

    static func fetch(tokenURL: URL, room: String, username: String) async throws -> LiveKitConnectionDetails {
        var request = URLRequest(url: tokenURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 12
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "room": room,
            "username": username
        ])

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw LiveKitTokenError.server(networkMessage(error, tokenURL: tokenURL))
        }

        guard let http = response as? HTTPURLResponse else {
            throw LiveKitTokenError.badResponse
        }

        if !(200..<300).contains(http.statusCode) {
            if let body = try? JSONDecoder().decode(ErrorBody.self, from: data), !body.error.isEmpty {
                throw LiveKitTokenError.server(body.error)
            }
            throw LiveKitTokenError.server("Token server returned HTTP \(http.statusCode).")
        }

        do {
            return try JSONDecoder().decode(LiveKitConnectionDetails.self, from: data)
        } catch {
            throw LiveKitTokenError.badResponse
        }
    }

    static func message(for error: Error, tokenURL: String) -> String {
        let base = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        let lower = tokenURL.lowercased()
        if lower.contains("127.0.0.1") || lower.contains("localhost") {
            return base + " Phones can't reach 127.0.0.1 on your Mac — use the Mac's LAN address."
        }
        return base
    }

    private static func networkMessage(_ error: Error, tokenURL: URL) -> String {
        let base = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        let host = tokenURL.host?.lowercased() ?? ""
        if host == "127.0.0.1" || host == "localhost" {
            return base + " Phones can't reach 127.0.0.1 on your Mac — use the Mac's LAN address."
        }
        return "Can't reach the call token server. \(base)"
    }
}
