import ARKit
import Foundation
import Network
import QuartzCore
import UIKit

/// Tracks the phone's 6-DoF pose with ARKit world tracking (camera + IMU) and
/// streams it to the Mac over UDP, one small JSON packet per camera frame.
final class Tracker: NSObject, ObservableObject, ARSessionDelegate {
    @Published var status = "Starting"
    @Published var trackingOK = false
    @Published var rate = 0
    /// What the Mac says is wrong while you drive: "reach", "limit", or "" when all is well.
    @Published var warning = ""
    /// Name of the Mac that found this phone by itself, or "" when the address was typed.
    @Published var peer = ""
    private var listener: NWListener?

    /// Set by the UI. Engaged = the arm follows the phone; released = the arm holds.
    var engaged = false
    /// 1 = fully open, 0 = fully closed.
    var gripOpen: Double = 1
    /// Fine control: the arm moves a fifth as far for the same hand motion.
    var fine = false
    /// Bumped on every press of Home. A counter survives a lost or repeated packet.
    var homeCount = 0
    private var lastBuzz = 0.0

    private let session = ARSession()
    private var connection: NWConnection?
    private var target = ""
    private var sent = 0
    private var rateStart = CACurrentMediaTime()

    func start() {
        guard ARWorldTrackingConfiguration.isSupported else {
            status = "ARKit world tracking is not available here (simulator?)"
            return
        }
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity  // y is up; heading is set by the Mac on each engage
        session.delegate = self
        session.run(config)
    }

    /// Wait for a Mac to announce itself. On the phone's own hotspot the Mac's
    /// router is this phone, so `stereo-hand phone` reaches us with no address
    /// typed; we then answer on the very connection it opened.
    func startListening() {
        guard listener == nil else { return }
        let params = NWParameters.udp
        params.allowLocalEndpointReuse = true
        guard let l = try? NWListener(using: params, on: 5007) else { return }
        l.newConnectionHandler = { [weak self] c in
            c.start(queue: .global(qos: .userInteractive))
            self?.awaitHello(c)
        }
        l.start(queue: .global(qos: .userInteractive))
        listener = l
    }

    private func awaitHello(_ c: NWConnection) {
        c.receiveMessage { [weak self] data, _, _, error in
            guard let self, error == nil else { c.cancel(); return }
            if let data,
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               obj["armpilot"] != nil {
                let name = obj["name"] as? String ?? "Mac"
                DispatchQueue.main.async { self.adopt(c, name: name) }
            } else {
                self.awaitHello(c)
            }
        }
    }

    private func adopt(_ c: NWConnection, name: String) {
        guard connection !== c else { return }
        connection?.cancel()
        connection = c
        target = "paired"
        peer = name
        listen(on: c)
    }

    func connect(host: String, port: UInt16) {
        let key = "\(host):\(port)"
        guard key != target, !host.isEmpty, let p = NWEndpoint.Port(rawValue: port) else { return }
        target = key
        peer = ""
        connection?.cancel()
        let c = NWConnection(host: NWEndpoint.Host(host), port: p, using: .udp)
        c.start(queue: .global(qos: .userInteractive))
        connection = c
        listen(on: c)
    }

    /// The Mac answers on the same socket, ten times a second, with what to warn about.
    private func listen(on c: NWConnection) {
        c.receiveMessage { [weak self] data, _, _, error in
            guard let self, error == nil, c === self.connection else { return }
            if let data,
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let warn = obj["warn"] as? String {
                DispatchQueue.main.async { self.apply(warning: warn) }
            }
            self.listen(on: c)
        }
    }

    private func apply(warning warn: String) {
        let now = CACurrentMediaTime()
        if warn != warning {
            warning = warn
            if !warn.isEmpty {
                UINotificationFeedbackGenerator().notificationOccurred(.warning)
                lastBuzz = now
            }
        } else if !warn.isEmpty, now - lastBuzz > 0.6 {
            // still against the stop: keep nudging, gently
            UIImpactFeedbackGenerator(style: .heavy).impactOccurred()
            lastBuzz = now
        }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let cam = frame.camera
        let state: String
        switch cam.trackingState {
        case .normal: state = "normal"
        case .notAvailable: state = "none"
        case .limited(let reason):
            switch reason {
            case .excessiveMotion: state = "limited: moving too fast"
            case .insufficientFeatures: state = "limited: camera sees too little detail"
            case .initializing: state = "limited: starting up"
            case .relocalizing: state = "limited: relocalising"
            @unknown default: state = "limited"
            }
        @unknown default: state = "unknown"
        }
        let m = cam.transform
        let c0 = m.columns.0, c1 = m.columns.1, c2 = m.columns.2, p = m.columns.3
        // rot is the camera's x, y, z axes in world coordinates, as columns
        let json = String(
            format: "{\"t\":%.4f,\"pos\":[%.5f,%.5f,%.5f],\"rot\":[[%.5f,%.5f,%.5f],[%.5f,%.5f,%.5f],[%.5f,%.5f,%.5f]],\"engaged\":%@,\"grip_open\":%.3f,\"grip\":%d,\"fine\":%@,\"home\":%d,\"tracking\":\"%@\"}",
            frame.timestamp, p.x, p.y, p.z,
            c0.x, c1.x, c2.x, c0.y, c1.y, c2.y, c0.z, c1.z, c2.z,
            engaged ? "true" : "false", gripOpen, gripOpen < 0.5 ? 1 : 0,
            fine ? "true" : "false", homeCount, state)
        connection?.send(content: json.data(using: .utf8), completion: .idempotent)

        sent += 1
        let now = CACurrentMediaTime()
        if now - rateStart >= 1 {
            rate = Int(Double(sent) / (now - rateStart))
            sent = 0
            rateStart = now
        }
        let ok = state == "normal"
        if ok != trackingOK { trackingOK = ok }
        if state != status { status = state }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        status = "ARKit error: \(error.localizedDescription)"
    }
}
