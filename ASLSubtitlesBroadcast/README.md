# ASL Subtitles Broadcast Upload Extension

Scaffold for **Control Center → Screen Broadcasting → ASL Subtitles** while FaceTime is open.

## Xcode setup (complete once on your Mac)

1. Open `ASLSubtitles.xcodeproj`.
2. **File → New → Target → Broadcast Upload Extension** (and optional Broadcast Setup UI).
3. Product Name: `ASLSubtitlesBroadcast` · Bundle ID: `com.officiallygoob.aslsubtitles.broadcast`.
4. Replace the generated `SampleHandler.swift` with the file in this folder (or add these sources to the target).
5. Signing: same Team as the app (`officiallygoob`).
6. Add **App Group** `group.com.officiallygoob.aslsubtitles` to **both** the app and the extension.
7. Embed the extension in the app target’s **Frameworks, Libraries, and Embedded Content**.

The main app polls the App Group for `broadcast.active` / `broadcast.liveCaption`. Heavy Vision recognition prefers **ScreenCaptureKit Call Mode** in-process; the extension keeps the broadcast session alive over FaceTime.
