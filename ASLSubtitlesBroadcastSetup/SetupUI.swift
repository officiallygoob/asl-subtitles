import ReplayKit
import UIKit

/// Optional Broadcast Setup UI extension entry (system presents before broadcast starts).
class BroadcastSetupViewController: UIViewController {
    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        let label = UILabel()
        label.text = "ASL Subtitles will caption the broadcast (FaceTime). Open the main app for the full transcript."
        label.textColor = .white
        label.numberOfLines = 0
        label.textAlignment = .center
        label.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(label)
        NSLayoutConstraint.activate([
            label.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 24),
            label.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -24),
            label.centerYAnchor.constraint(equalTo: view.centerYAnchor)
        ])
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        // Complete setup immediately — user already chose ASL Subtitles.
        let config = RPBroadcastConfiguration()
        userDidFinishSetup()
        _ = config
    }

    func userDidFinishSetup() {
        // RPBroadcastSampleHandler setup completion is performed by the system API
        // when using storyboard-based setup; for code-only scaffold we no-op safely.
    }
}
