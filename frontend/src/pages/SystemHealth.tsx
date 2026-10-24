import React from "react";
import { useApp } from "../store";
import { useProviderData, useClock } from "../hooks";
import { Panel, StatusPill, pipelineStateTone } from "../components/common";

const COMPONENT_LABELS: [string, string][] = [
  ["ETHERNET", "Ethernet Receiver"], ["ARRAY_PROCESSING", "Array Processing"],
  ["BEAMFORMER", "Beamformer"], ["DOA", "DOA"], ["SEGFORMER", "SegFormer"],
  ["POSTPROCESSING", "Postprocessing"], ["CLASSIFIER", "Classifier"], ["TRACKER", "Kalman Tracker"],
];

export function SystemHealthPage() {
  const { mode, provider } = useApp();
  const { data: status } = useProviderData((p) => p.getStatus());
  const clock = useClock();
  const connState = provider.getConnectionState();
  const overallOk = mode === "demo" ? true : connState === "live";

  return (
    <div className="flex-col gap-3">
      <Panel title="System Health">
        <div className="flex items-center gap-3" style={{ marginBottom: 16 }}>
          <StatusPill tone={overallOk ? "ok" : "err"} pulse={overallOk}>
            {overallOk ? "OPERATIONAL" : "BACKEND NOT CONNECTED"}
          </StatusPill>
          {mode === "demo" && <StatusPill tone="warn">DATA SOURCE: DEMO / SIMULATED</StatusPill>}
          {mode === "live" && <StatusPill tone={connState === "live" ? "ok" : "err"}>DATA SOURCE: {connState === "live" ? "LIVE BACKEND" : "OFFLINE"}</StatusPill>}
        </div>

        <div className="grid" style={{ gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 16 }}>
          <StatBox label="Uptime" value={formatUptime(status?.system.uptime_s ?? 0)} />
          <StatBox label="Last Update" value={clock.toLocaleTimeString(undefined, { hour12: false })} />
          <StatBox label="Frames Processed" value={String(status?.system.n_frames_processed ?? 0)} />
          <StatBox label="Packets Received" value={String(status?.network.packets_received ?? 0)} />
        </div>

        <table className="data-table">
          <thead><tr><th>Component</th><th>State</th></tr></thead>
          <tbody>
            <tr>
              <td>Signal Generator</td>
              <td><StatusPill tone={mode === "demo" ? "ok" : (status ? "ok" : "off")}>{mode === "demo" ? "READY (simulated)" : status ? "READY" : "UNAVAILABLE"}</StatusPill></td>
            </tr>
            {COMPONENT_LABELS.map(([key, label]) => (
              <tr key={key}>
                <td>{label}</td>
                <td><StatusPill tone={pipelineStateTone(status?.pipeline?.[key] ?? "UNAVAILABLE")}>{status?.pipeline?.[key] ?? "UNAVAILABLE"}</StatusPill></td>
              </tr>
            ))}
            <tr><td>Hydrophone Array</td><td><StatusPill tone="ok">{status?.sensors?.filter((s) => s.online).length ?? 0}/{status?.array.n_hydrophones ?? 8}</StatusPill></td></tr>
            <tr><td>REST API</td><td><StatusPill tone={status ? "ok" : "off"}>{status ? "READY" : "UNAVAILABLE"}</StatusPill></td></tr>
            <tr><td>WebSocket</td><td><StatusPill tone={connState === "live" ? "ok" : mode === "demo" ? "off" : "err"}>{mode === "demo" ? "N/A (demo)" : connState.toUpperCase()}</StatusPill></td></tr>
          </tbody>
        </table>
      </Panel>
    </div>
  );
}

function StatBox({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ padding: 12, background: "var(--bg-inset)", borderRadius: 6 }}>
      <div className="text-tertiary" style={{ fontSize: 10, textTransform: "uppercase", marginBottom: 3 }}>{label}</div>
      <div className="mono" style={{ fontWeight: 700, fontSize: 14 }}>{value}</div>
    </div>
  );
}

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}
