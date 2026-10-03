import ARKit
import Foundation
import Network
import QuartzCore

/// Tracks the phone's 6-DoF pose with ARKit world tracking (camera + IMU) and
/// streams it to the Mac over UDP, one small JSON packet per camera frame.
final class Tracker: NSObject, ObservableObject, ARSessionDelegate {
    @Published var status = "Starting"
    @Published var trackingOK = false
    @Published var rate = 0

    /// Set by the UI. Engaged = the arm follows the phone; released = the arm holds.
    var engaged = false
    var gripClosed = false

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

    func connect(host: String, port: UInt16) {
        let key = "\(host):\(port)"
        guard key != target, !host.isEmpty, let p = NWEndpoint.Port(rawValue: port) else { return }
        target = key
        connection?.cancel()
        let c = NWConnection(host: NWEndpoint.Host(host), port: p, using: .udp)
        c.start(queue: .global(qos: .userInteractive))
        connection = c
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
            format: "{\"t\":%.4f,\"pos\":[%.5f,%.5f,%.5f],\"rot\":[[%.5f,%.5f,%.5f],[%.5f,%.5f,%.5f],[%.5f,%.5f,%.5f]],\"engaged\":%@,\"grip\":%d,\"tracking\":\"%@\"}",
            frame.timestamp, p.x, p.y, p.z,
            c0.x, c1.x, c2.x, c0.y, c1.y, c2.y, c0.z, c1.z, c2.z,
            engaged ? "true" : "false", gripClosed ? 1 : 0, state)
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
