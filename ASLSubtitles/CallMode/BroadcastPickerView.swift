import SwiftUI
import ReplayKit

/// System broadcast picker targeting the ASL Subtitles Broadcast Upload extension.
struct BroadcastPickerView: UIViewRepresentable {
    var preferredExtension: String = "com.officiallygoob.aslsubtitles.broadcast"

    func makeUIView(context: Context) -> RPSystemBroadcastPickerView {
        let picker = RPSystemBroadcastPickerView(frame: CGRect(x: 0, y: 0, width: 60, height: 60))
        picker.preferredExtension = preferredExtension
        picker.showsMicrophoneButton = false
        if let button = picker.subviews.first(where: { $0 is UIButton }) as? UIButton {
            button.tintColor = .white
        }
        return picker
    }

    func updateUIView(_ uiView: RPSystemBroadcastPickerView, context: Context) {
        uiView.preferredExtension = preferredExtension
    }
}
