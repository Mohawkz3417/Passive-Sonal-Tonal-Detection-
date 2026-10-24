import React from "react";
import { useApp } from "../store";
import { useProviderData } from "../hooks";
import { Panel, StatusPill } from "../components/common";
import { Waterfall } from "../components/visualizations";

export function SpectrogramPage() {
  const { mode } = useApp();
  const { data: spectrogram } = useProviderData((p) => p.getSpectrogram(), 800);
  const { data: detection } = useProviderData((p) => p.getLatestDetection());

  return (
    <div className="flex-col gap-3">
      <Panel title="Live Spectrogram / Waterfall" actions={<StatusPill tone={mode === "demo" ? "warn" : "ok"}>DATA SOURCE: {mode.toUpperCase()}</StatusPill>}>
        <Waterfall data={spectrogram ?? null} height={320} />
      </Panel>
      <div className="grid" style={{ gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
        <InfoTile label="Primary" value={detection?.frequency_hz ? `${detection.frequency_hz.toFixed(0)} Hz` : "—"} />
        <InfoTile label="Harmonics" value={detection?.n_harmonics ? `${detection.n_harmonics} detected` : "—"} />
        <InfoTile label="Bandwidth" value={detection?.bandwidth_hz ? `${detection.bandwidth_hz.toFixed(1)} Hz` : "—"} />
        <InfoTile label="Threshold" value={spectrogram?.seg_threshold?.toFixed(2) ?? "—"} />
      </div>
    </div>
  );
}

function InfoTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="panel" style={{ padding: 14 }}>
      <div style={{ fontSize: 10, color: "var(--text-tertiary)", letterSpacing: "0.06em", marginBottom: 4 }}>{label.toUpperCase()}</div>
      <div className="mono" style={{ fontSize: 15, fontWeight: 700 }}>{value}</div>
    </div>
  );
}
