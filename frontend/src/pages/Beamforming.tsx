import React from "react";
import { useProviderData } from "../hooks";
import { Panel, MetricTile } from "../components/common";
import { BeamChart } from "../components/visualizations";

export function BeamformingPage() {
  const { data: beam } = useProviderData((p) => p.getBeamSpectrum(), 1000);
  const { data: status } = useProviderData((p) => p.getStatus());

  return (
    <div className="flex-col gap-3">
      <Panel title="Beam Power Spectrum">
        <BeamChart data={beam} />
      </Panel>
      <div className="grid" style={{ gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
        <MetricTile label="Peak Bearing" value={beam?.peak_angle_deg?.toFixed(1) ?? "—"} unit="°" tone="ok" />
        <MetricTile label="Scan Range" value="-90 → +90" unit="°" />
        <MetricTile label="Scan Resolution" value="2" unit="°" />
        <MetricTile label="Beamformer" value="Delay-and-Sum" />
      </div>
      <Panel title="Array Geometry">
        <div className="grid" style={{ gridTemplateColumns: "repeat(3, 1fr)", gap: 10 }}>
          <StatRow label="Hydrophones" value={String(status?.array.n_hydrophones ?? 8)} />
          <StatRow label="Spacing" value={`${status?.array.spacing_m ?? 0.2} m`} />
          <StatRow label="Sound Speed" value={`${status?.array.sound_speed_mps ?? 1500} m/s`} />
        </div>
      </Panel>
    </div>
  );
}

function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ padding: "10px 14px", background: "var(--bg-inset)", borderRadius: 6 }}>
      <div className="text-tertiary" style={{ fontSize: 10.5, textTransform: "uppercase", marginBottom: 3 }}>{label}</div>
      <div className="mono" style={{ fontWeight: 700 }}>{value}</div>
    </div>
  );
}
