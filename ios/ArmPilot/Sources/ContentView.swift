import SwiftUI

struct ContentView: View {
    @StateObject private var tracker = Tracker()
    @AppStorage("host") private var host = ""
    @AppStorage("port") private var port = 5006
    @State private var engaged = false
    @State private var gripClosed = false
    @State private var touchStart: Date?

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                TextField("Mac IP address", text: $host)
                    .keyboardType(.numbersAndPunctuation)
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
                Text("Tracking \(tracker.status)").font(.footnote)
                Spacer()
                Text("\(tracker.rate)/s").font(.footnote.monospacedDigit())
            }
            .foregroundStyle(.secondary)

            Button {
                gripClosed.toggle()
                tracker.gripClosed = gripClosed
            } label: {
                Text(gripClosed ? "GRIPPER CLOSED" : "GRIPPER OPEN")
                    .font(.title3.bold())
                    .frame(maxWidth: .infinity, minHeight: 64)
            }
            .buttonStyle(.borderedProminent)
            .tint(gripClosed ? .red : .gray)

            ZStack {
                RoundedRectangle(cornerRadius: 24)
                    .fill(engaged ? Color.green.opacity(0.85) : Color(.secondarySystemBackground))
                VStack(spacing: 8) {
                    Text(engaged ? "MOVING" : "HOLD TO MOVE").font(.largeTitle.bold())
                    Text(engaged ? "Lift to let go: the arm holds still"
                                 : "Hold: the arm follows the phone\nTap: open or close the gripper")
                        .multilineTextAlignment(.center)
                        .font(.callout)
                }
                .foregroundStyle(engaged ? .white : .primary)
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in
                        if touchStart == nil {
                            touchStart = Date()
                            setEngaged(true)
                        }
                    }
                    .onEnded { v in
                        let quick = Date().timeIntervalSince(touchStart ?? .distantPast) < 0.25
                        let still = hypot(v.translation.width, v.translation.height) < 20
                        touchStart = nil
                        setEngaged(false)
                        if quick && still {
                            gripClosed.toggle()
                            tracker.gripClosed = gripClosed
                        }
                    }
            )

            Text("Hold the phone screen-up, top edge pointing where the arm should reach. The back camera needs to see the room.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true
            tracker.start()
            tracker.connect(host: host, port: UInt16(clamping: port))
        }
        .onChange(of: host) { _, h in tracker.connect(host: h, port: UInt16(clamping: port)) }
        .onChange(of: port) { _, p in tracker.connect(host: host, port: UInt16(clamping: p)) }
    }

    private func setEngaged(_ on: Bool) {
        engaged = on
        tracker.engaged = on
        UIImpactFeedbackGenerator(style: on ? .medium : .light).impactOccurred()
    }
}
