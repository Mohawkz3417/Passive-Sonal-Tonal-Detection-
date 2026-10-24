import React, { useState } from "react";
import { useApp } from "../store";
import { useProviderData } from "../hooks";
import { Panel, MetricTile, StatusPill } from "../components/common";
import { PPIDisplay } from "../components/PPIDisplay";
import { BeamChart } from "../components/visualizations";
import { PipelineStages, EventLog } from "../components/PipelineAndEvents";
import { ContactDetailDrawer } from "../components/ContactsUI";
import { DemoControlDrawer } from "../components/DemoControls";

export function Dashboard() {
  const { provider, mode } = useApp();
  const { data: status } = useProviderData((p) => p.getStatus());
  const { data: contacts } = useProviderData((p) => p.getContacts());
  const { data: detection } = useProviderData((p) => p.getLatestDetection());
  const { data: events } = useProviderData((p) => p.getEvents());
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const list = contacts ?? [];
  const selected = list.find((c) => c.id === selectedId) ?? null;
  const activeContacts = list.length;
  const confirmedTracks = list.filter((c) => c.status === "CONFIRMED").length;

  return (
    <div className="flex-col gap-3">
      {mode === "demo" && <DemoControlDrawer />}

      <div className="grid" style={{ gridTemplateColumns: "repeat(7, 1fr)", gap: 10 }}>
        <MetricTile label="System" value={status?.pipeline?.SEGFORMER === "RUNNING" || status?.pipeline?.SEGFORMER === "READY" ? "OPERATIONAL" : "OFFLINE"}
                    tone={status ? "ok" : "off"} />
        <MetricTile label="Active Contacts" value={String(activeContacts).padStart(2, "0")} />
        <MetricTile label="Tracks" value={String(confirmedTracks).padStart(2, "0")} tone="ok" />
        <MetricTile label="Array" value={`${status?.sensors?.filter((s) => s.online).length ?? 0}/${status?.array?.n_hydrophones ?? 8}`} />
        <MetricTile label="Primary Bearing" value={detection?.estimated_bearing_deg?.toFixed(1) ?? "—"} unit="°" />
        <MetricTile label="Primary Frequency" value={detection?.frequency_hz?.toFixed(0) ?? "—"} unit="Hz" />
        <MetricTile label="SNR" value={detection?.measured_snr_db?.toFixed(1) ?? "—"} unit="dB"
                    tone={detection?.measured_snr_db != null && detection.measured_snr_db < -8 ? "warn" : "ok"} />
      </div>

      <div className="grid" style={{ gridTemplateColumns: "1.1fr 1fr", gap: 12 }}>
        <Panel title="Sonar / Bearing Display" actions={<StatusPill tone={provider.getConnectionState() === "live" ? "ok" : "off"} pulse>{mode.toUpperCase()}</StatusPill>}>
          <PPIDisplay contacts={list} selectedId={selectedId} onSelect={setSelectedId} sweeping />
          {selected && (
            <div className="flex justify-center gap-3 mono" style={{ fontSize: 12, marginTop: 10 }}>
              <span style={{ fontWeight: 700 }}>{selected.id}</span>
              <span>{selected.bearing_deg?.toFixed(0)}°</span>
              <span>{selected.frequency_hz?.toFixed(0)} Hz</span>
              <span>{selected.snr_db?.toFixed(1)} dB</span>
              <span className="text-accent">{selected.classification}</span>
              <span>{selected.confidence !== null ? `${(selected.confidence * 100).toFixed(0)}%` : ""}</span>
            </div>
          )}
        </Panel>

        <div className="flex-col gap-3">
          <Panel title="Beamforming">
            <BeamChart data={detection?.beam_spectrum ?? null} />
          </Panel>
          <Panel title="Processing Pipeline">
            <PipelineStages pipeline={status?.pipeline} />
          </Panel>
        </div>
      </div>

      <Panel title="System Event Log">
        <EventLog events={events ?? []} />
      </Panel>

      <ContactDetailDrawer contact={selected} onClose={() => setSelectedId(null)} />
    </div>
  );
}
