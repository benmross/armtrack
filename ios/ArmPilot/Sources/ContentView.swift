import SwiftUI

struct ContentView: View {
    @StateObject private var tracker = Tracker()
    @AppStorage("host") private var host = ""
    @AppStorage("port") private var port = 5006
    @State private var engaged = false
    @State private var gripOpen = 1.0          // 1 open, 0 closed
    @State private var gripAtTouch: Double?
    @State private var sliding = false
    @State private var fine = false

    /// Thumb travel, in points, from fully open to fully closed.
    private let travel = 240.0
    /// Movement ignored before a slide counts, so holding still never changes the grip.
    private let deadzone = 14.0

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                TextField("Mac address (not needed on hotspot)", text: $host)
                    .keyboardType(.URL)
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
                    .textFieldStyle(.roundedBorder)
                TextField("Port", value: $port, format: .number.grouping(.never))
                    .keyboardType(.numberPad)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 80)
            }
            HStack {
                Circle().fill(tracker.trackingOK ? .green : .orange).frame(width: 10, height: 10)
                Text(tracker.peer.isEmpty ? "Tracking \(tracker.status)"
                                          : "Paired with \(tracker.peer) · tracking \(tracker.status)")
                    .font(.footnote)
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                Spacer()
                Text("\(tracker.rate)/s").font(.footnote.monospacedDigit())
            }
            .foregroundStyle(.secondary)

            HStack(spacing: 10) {
                Button {
                    fine.toggle()
                    tracker.fine = fine
                    UIImpactFeedbackGenerator(style: .light).impactOccurred()
                } label: {
                    VStack(spacing: 2) {
                        Text(fine ? "FINE" : "NORMAL").font(.headline)
                        Text(fine ? "arm moves 0.5x" : "arm moves 2.5x").font(.caption2)
                    }
                    .frame(maxWidth: .infinity, minHeight: 52)
                }
                .buttonStyle(.borderedProminent)
                .tint(fine ? .blue : .gray)

                Button {
                    tracker.homeCount += 1
                    UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                } label: {
                    VStack(spacing: 2) {
                        Text("HOME").font(.headline)
                        Text("back to rest").font(.caption2)
                    }
                    .frame(maxWidth: .infinity, minHeight: 52)
                }
                .buttonStyle(.bordered)
            }

            HStack(spacing: 12) {
                ZStack {
                    RoundedRectangle(cornerRadius: 24)
                        .fill(padColor)
                    VStack(spacing: 8) {
                        Text(padTitle).font(.largeTitle.bold())
                        Text(engaged ? "Slide up to open, down to close\nLift to let go: the arm holds still"
                                     : "Hold: the arm follows the phone\nSlide up or down: open or close the gripper")
                            .multilineTextAlignment(.center)
                            .font(.callout)
                    }
                    .foregroundStyle(engaged ? .white : .primary)
                    .padding(.horizontal, 8)
                }
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { v in
                            if gripAtTouch == nil {
                                gripAtTouch = gripOpen
                                sliding = false
                                setEngaged(true)
                            }
                            let dy = Double(v.translation.height)
                            if !sliding && abs(dy) > deadzone { sliding = true }
                            guard sliding, let start = gripAtTouch else { return }
                            // past the deadzone, thumb travel maps straight onto the gripper: up opens
                            let moved = dy > 0 ? dy - deadzone : dy + deadzone
                            setGrip(start - moved / travel)
                        }
                        .onEnded { _ in
                            gripAtTouch = nil
                            sliding = false
                            setEngaged(false)
                        }
                )

                // gripper gauge: full = open, empty = closed
                VStack(spacing: 6) {
                    Text("OPEN").font(.caption2.bold()).foregroundStyle(.secondary)
                    GeometryReader { geo in
                        ZStack(alignment: .bottom) {
                            RoundedRectangle(cornerRadius: 10).fill(Color(.secondarySystemBackground))
                            RoundedRectangle(cornerRadius: 10)
                                .fill(gripOpen < 0.08 ? Color.red : Color.orange)
                                .frame(height: max(6, geo.size.height * gripOpen))
                        }
                    }
                    .frame(width: 34)
                    Text("CLOSED").font(.caption2.bold()).foregroundStyle(.secondary)
                    Text("\(Int((gripOpen * 100).rounded()))%").font(.footnote.monospacedDigit())
                }
            }

            Text("Hold the phone screen-up, top edge pointing where the arm should reach. The back camera needs to see the room.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true
            tracker.gripOpen = gripOpen
            tracker.start()
            tracker.startListening()
            tracker.connect(host: host, port: UInt16(clamping: port))
        }
        .onChange(of: host) { _, h in tracker.connect(host: h, port: UInt16(clamping: port)) }
        .onChange(of: port) { _, p in tracker.connect(host: host, port: UInt16(clamping: p)) }
    }

    private var warned: Bool { engaged && !tracker.warning.isEmpty }

    private var padColor: Color {
        if warned { return Color.red.opacity(0.85) }
        return engaged ? Color.green.opacity(0.85) : Color(.secondarySystemBackground)
    }

    private var padTitle: String {
        if warned { return tracker.warning == "reach" ? "OUT OF REACH" : "JOINT LIMIT" }
        return engaged ? "MOVING" : "HOLD TO MOVE"
    }

    private func setGrip(_ value: Double) {
        let g = min(1, max(0, value))
        // one tick as the gripper reaches either end, so you can feel it without looking
        if (g == 0 || g == 1) && g != gripOpen {
            UIImpactFeedbackGenerator(style: .rigid).impactOccurred()
        }
        gripOpen = g
        tracker.gripOpen = g
    }

    private func setEngaged(_ on: Bool) {
        engaged = on
        tracker.engaged = on
        UIImpactFeedbackGenerator(style: on ? .medium : .light).impactOccurred()
    }
}
