import Foundation

/// On-device bigram prior over glosses for top-k logit rerank.
/// Static conversational templates — no network.
enum GlossBigramPrior {
    /// prev → next → weight
    private static let table: [String: [String: Double]] = {
        let templates: [[String]] = [
            ["HELLO", "HOW", "YOU"],
            ["HI", "ME", "FINE"],
            ["HOW", "YOU"],
            ["YOU", "OK"],
            ["ME", "FINE"],
            ["ME", "GOOD"],
            ["ME", "TIRED"],
            ["ME", "HUNGRY"],
            ["ME", "WANT", "FOOD"],
            ["ME", "NEED", "HELP"],
            ["WHAT", "YOU", "WANT"],
            ["WHERE", "YOU", "GO"],
            ["WHY", "YOU", "LEAVE"],
            ["PLEASE", "HELP"],
            ["THANKS", "YOU"],
            ["YES", "ME", "UNDERSTAND"],
            ["NO", "ME", "DONT-KNOW"],
            ["ME", "GO", "HOME"],
            ["SEE", "YOU", "LATER"],
            ["ME", "LOVE", "YOU"],
            ["YOU", "LIKE", "MOVIE"],
            ["ME", "EAT", "FOOD"],
            ["ME", "CALL", "YOU"],
            ["WHAT", "YOUR", "NAME"],
            ["PLEASE", "SLOW"],
            ["AGAIN", "PLEASE"],
            ["CAN", "YOU", "HELP"],
            ["ME", "FINISH"],
            ["GO", "BATHROOM"],
            ["ME", "AND", "YOU"],
            ["ABOUT", "WHAT"],
            ["PARTY", "WHEN"],
            ["CHRISTMAS", "PARTY"],
        ]
        var counts: [String: [String: Double]] = [:]
        for tmpl in templates {
            for i in 0..<(tmpl.count - 1) {
                let a = tmpl[i], b = tmpl[i + 1]
                var row = counts[a] ?? [:]
                row[b, default: 0] += 1
                counts[a] = row
            }
        }
        var out: [String: [String: Double]] = [:]
        for (prev, row) in counts {
            let s = row.values.reduce(0, +)
            guard s > 0 else { continue }
            out[prev] = row.mapValues { $0 / s }
        }
        return out
    }()

    static func rerank(probs: [Double], labels: [String], prevGloss: String?, k: Int = 5, priorWeight: Double = 0.25) -> Int {
        guard !probs.isEmpty, probs.count == labels.count else {
            return probs.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
        }
        let indexed = probs.enumerated().sorted { $0.element > $1.element }
        let top = Array(indexed.prefix(min(k, indexed.count)))
        guard let prev = prevGloss, let row = table[prev], !row.isEmpty else {
            return top[0].offset
        }
        var bestIdx = top[0].offset
        var bestScore = -Double.greatestFiniteMagnitude
        for (idx, p) in top {
            let g = labels[idx]
            let pb = row[g] ?? 1e-3
            let score = (1.0 - priorWeight) * log(max(p, 1e-12)) + priorWeight * log(max(pb, 1e-12))
            if score > bestScore {
                bestScore = score
                bestIdx = idx
            }
        }
        return bestIdx
    }
}
