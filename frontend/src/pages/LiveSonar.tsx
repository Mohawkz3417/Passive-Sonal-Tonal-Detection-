import React, { useState } from "react";
import { useApp } from "../store";
import { useProviderData } from "../hooks";
import { Panel, StatusPill } from "../components/common";
import { PPIDisplay } from "../components/PPIDisplay";
import { PipelineStages } from "../components/PipelineAndEvents";
import { ContactDetailDrawer } from "../components/ContactsUI";

export function LiveSonar() {
  const { mode, provider } = useApp();
  const { data: status } = useProviderData((p) => p.getStatus());
  const { data: contacts } = useProviderData((p) => p.getContacts());
  const { data: detection } = useProviderData((p) => p.getLatestDetection());
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const list = contacts ?? [];
  const selected = list.find((c) => c.id === selectedId) ?? null;

  return (
    <div className="grid" style={{ gridTemplateColumns: "1.4fr 1fr", gap: 14 }}>
      <Panel title="Live Sonar View">
        <PPIDisplay contacts={list} selectedId={selectedId} onSelect={setSelectedId} sweeping />
      </Panel>

      <div className="flex-col gap-3">
        <Panel title="Acquisition">
          <div className="flex-col gap-2">
            <Row label="Acquisition" value={<StatusPill tone={status?.input.connected ? "ok" : "off"} pulse={status?.input.connected}>{status?.input.connected ? "ACTIVE" : "IDLE"}</StatusPill>} />
            <Row label="Data source" value={<span className="mono">{mode.toUpperCase()}</span>} />
            <Row label="Channels" value={<span className="mono">{status?.array.n_hydrophones ?? 8}</span>} />
            <Row label="Sampling" value={<span className="mono">{((status?.network.sample_rate_hz ?? 16000) / 1000).toFixed(0)} kHz</span>} />
            <Row label="Chunk" value={<span className="mono">4.0 s</span>} />
            <Row label="Connection" value={<span className="mono">{provider.getConnectionState().toUpperCase()}</span>} />
          </div>
        </Panel>

        <Panel title="Live Processing Stages">
          <PipelineStages pipeline={status?.pipeline} />
        </Panel>

        <Panel title="Current Detection">
          {detection?.detected ? (
            <div className="flex-col gap-1 mono" style={{ fontSize: 12.5 }}>
              <Row label="Bearing" value={`${detection.estimated_bearing_deg?.toFixed(1)}°`} />
              <Row label="Frequency" value={`${detection.frequency_hz?.toFixed(0)} Hz`} />
              <Row label="SNR" value={`${detection.measured_snr_db?.toFixed(1)} dB`} />
              <Row label="Classification" value={detection.classification ?? "—"} />
              <Row label="Confidence" value={detection.classification_confidence !== null ? `${(detection.classification_confidence * 100).toFixed(0)}%` : "—"} />
            </div>
          ) : <div className="text-tertiary" style={{ fontSize: 12.5 }}>No active detection.</div>}
        </Panel>
      </div>
      <ContactDetailDrawer contact={selected} onClose={() => setSelectedId(null)} />
    </div>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between" style={{ fontSize: 12 }}>
      <span className="text-secondary">{label}</span>
      <span>{value}</span>
    </div>
  );
}
