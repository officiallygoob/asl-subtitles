import AVFoundation
import Combine
import UIKit

/// Manages AVCaptureSession lifecycle, camera flip, and frame delivery.
final class CameraManager: NSObject, ObservableObject {
    let session = AVCaptureSession()

    @Published private(set) var isRunning = false
    @Published private(set) var isFrontCamera = false
    @Published private(set) var permissionStatus: AVAuthorizationStatus = .notDetermined

    private let sessionQueue = DispatchQueue(label: "com.aslsubtitles.camera.session")
    private let videoOutput = AVCaptureVideoDataOutput()
    private var currentInput: AVCaptureDeviceInput?
    private weak var frameDelegate: AVCaptureVideoDataOutputSampleBufferDelegate?

    var onConfigurationError: ((Error) -> Void)?

    override init() {
        super.init()
        permissionStatus = AVCaptureDevice.authorizationStatus(for: .video)
    }

    func setFrameDelegate(_ delegate: AVCaptureVideoDataOutputSampleBufferDelegate) {
        frameDelegate = delegate
        videoOutput.setSampleBufferDelegate(delegate, queue: DispatchQueue(label: "com.aslsubtitles.camera.frames"))
    }

    func refreshPermissionStatus() {
        permissionStatus = AVCaptureDevice.authorizationStatus(for: .video)
    }

    func requestAccess(completion: @escaping (Bool) -> Void) {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        switch status {
        case .authorized:
            permissionStatus = .authorized
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    self?.permissionStatus = AVCaptureDevice.authorizationStatus(for: .video)
                    completion(granted)
                }
            }
        default:
            permissionStatus = status
            completion(false)
        }
    }

    func configureAndStart(preferFront: Bool = false) {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            do {
                try self.configureSession(preferFront: preferFront)
                if !self.session.isRunning {
                    self.session.startRunning()
                }
                DispatchQueue.main.async {
                    self.isRunning = self.session.isRunning
                    self.isFrontCamera = preferFront
                }
            } catch {
                DispatchQueue.main.async {
                    self.onConfigurationError?(error)
                }
            }
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if self.session.isRunning {
                self.session.stopRunning()
            }
            DispatchQueue.main.async {
                self.isRunning = false
            }
        }
    }

    func toggleCamera() {
        let nextFront = !isFrontCamera
        sessionQueue.async { [weak self] in
            guard let self else { return }
            do {
                try self.reconfigureInput(preferFront: nextFront)
                DispatchQueue.main.async {
                    self.isFrontCamera = nextFront
                }
            } catch {
                DispatchQueue.main.async {
                    self.onConfigurationError?(error)
                }
            }
        }
    }

    // MARK: - Private

    private func configureSession(preferFront: Bool) throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        session.sessionPreset = .high

        session.inputs.forEach { session.removeInput($0) }
        session.outputs.forEach { session.removeOutput($0) }

        try addCameraInput(preferFront: preferFront)

        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        guard session.canAddOutput(videoOutput) else {
            throw CameraError.cannotAddOutput
        }
        session.addOutput(videoOutput)

        if let connection = videoOutput.connection(with: .video) {
            if connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
            if connection.isVideoMirroringSupported {
                connection.isVideoMirrored = preferFront
            }
        }
    }

    private func reconfigureInput(preferFront: Bool) throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        if let currentInput {
            session.removeInput(currentInput)
            self.currentInput = nil
        }
        try addCameraInput(preferFront: preferFront)

        if let connection = videoOutput.connection(with: .video) {
            if connection.isVideoMirroringSupported {
                connection.isVideoMirrored = preferFront
            }
        }
    }

    private func addCameraInput(preferFront: Bool) throws {
        let position: AVCaptureDevice.Position = preferFront ? .front : .back
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position)
                ?? AVCaptureDevice.default(for: .video) else {
            throw CameraError.noCamera
        }
        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else {
            throw CameraError.cannotAddInput
        }
        session.addInput(input)
        currentInput = input
    }
}

enum CameraError: LocalizedError {
    case noCamera
    case cannotAddInput
    case cannotAddOutput

    var errorDescription: String? {
        switch self {
        case .noCamera: return "No camera available on this device."
        case .cannotAddInput: return "Could not attach the camera input."
        case .cannotAddOutput: return "Could not attach the camera output."
        }
    }
}
