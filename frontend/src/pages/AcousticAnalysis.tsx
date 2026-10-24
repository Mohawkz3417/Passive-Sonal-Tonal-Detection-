import React from "react";
import { useProviderData } from "../hooks";
import { Panel } from "../components/common";

function ConfigRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between" style={{ padding: "7px 0", borderBottom: "1px solid var(--border-soft)" }}>
      <span className="text-secondary" style={{ fontSize: 12 }}>{label}</span>
      <span className="mono" style={{ fontSize: 12, fontWeight: 600 }}>{value}</span>
    </div>
  );
}

export function AcousticAnalysisPage() {
  const { data: status } = useProviderData((p) => p.getStatus());

  return (
    <div className="grid" style={{ gridTemplateColumns: "repeat(3, 1fr)", gap: 14 }}>
      <Panel title="Acquisition">
        <ConfigRow label="Sampling Rate" value={`${((status?.network.sample_rate_hz ?? 16000) / 1000).toFixed(0)} kHz`} />
        <ConfigRow label="Channels" value={status?.array.n_hydrophones ?? 8} />
        <ConfigRow label="Chunk" value="4.0 s" />
      </Panel>
      <Panel title="Signal Processing">
        <ConfigRow label="Bandpass" value="50 – 4000 Hz" />
        <ConfigRow label="FFT" value="1024" />
        <ConfigRow label="Hop Length" value="256" />
        <ConfigRow label="PCEN" value="Enabled" />
      </Panel>
      <Panel title="Array">
        <ConfigRow label="Hydrophones" value={status?.array.n_hydrophones ?? 8} />
        <ConfigRow label="Spacing" value={`${status?.array.spacing_m ?? 0.2} m`} />
        <ConfigRow label="Sound Speed" value={`${status?.array.sound_speed_mps ?? 1500} m/s`} />
      </Panel>
      <Panel title="DOA">
        <ConfigRow label="Range" value="-90° → +90°" />
        <ConfigRow label="Resolution" value="2°" />
        <ConfigRow label="Method" value="Delay-and-Sum" />
      </Panel>
      <Panel title="Detection">
        <ConfigRow label="Model" value="SegFormer-B0" />
        <ConfigRow label="Post-processing" value="CFAR" />
        <ConfigRow label="Cleanup" value="Morphology" />
        <ConfigRow label="Grouping" value="Connected Components" />
      </Panel>
      <Panel title="Tracking">
        <ConfigRow label="Tracker" value="Kalman (1D freq)" />
        <ConfigRow label="Gate" value="15 Hz" />
        <ConfigRow label="Confirm Hits" value="3" />
        <ConfigRow label="Max Misses" value="4" />
      </Panel>
    </div>
  );
}
