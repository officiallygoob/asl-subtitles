import AVFoundation
import SwiftUI

struct PermissionView: View {
    let status: AVAuthorizationStatus
    let onRequest: () -> Void

    var body: some View {
        VStack(spacing: 28) {
            Spacer()

            Image(systemName: "hand.raised.circle.fill")
                .font(.system(size: 72))
                .foregroundStyle(.orange)
                .accessibilityHidden(true)

            VStack(spacing: 12) {
                Text("ASL Subtitles")
                    .font(.largeTitle.weight(.bold))
                    .foregroundStyle(.white)

                Text(message)
                    .font(.body)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.white.opacity(0.85))
                    .padding(.horizontal, 32)
            }

            if status == .notDetermined {
                Button(action: onRequest) {
                    Text("Enable Camera")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                }
                .buttonStyle(.borderedProminent)
                .tint(.orange)
                .padding(.horizontal, 40)
            } else {
                Button {
                    if let url = URL(string: UIApplication.openSettingsURLString) {
                        UIApplication.shared.open(url)
                    }
                } label: {
                    Text("Open Settings")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                }
                .buttonStyle(.borderedProminent)
                .tint(.orange)
                .padding(.horizontal, 40)
            }

            Text("On-device only · Nothing is uploaded")
                .font(.footnote)
                .foregroundStyle(.white.opacity(0.55))

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(
            LinearGradient(
                colors: [Color.black, Color(red: 0.12, green: 0.08, blue: 0.05)],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()
        )
    }

    private var message: String {
        switch status {
        case .notDetermined:
            return "Point the camera at a friend’s signing hands. We’ll show live English subtitles using on-device hand tracking — a limited starter vocabulary, not full ASL."
        case .denied, .restricted:
            return "Camera access is off. Enable it in Settings so ASL Subtitles can watch for signs on your device."
        default:
            return "Camera access is required for live subtitles."
        }
    }
}
