import React from "react";
import { useProviderData } from "../hooks";
import { Panel, MetricTile } from "../components/common";
import { ArrayGrid } from "../components/ArrayGrid";

export function ArrayStatusPage() {
  const { data: status } = useProviderData((p) => p.getStatus());
  const sensors = status?.sensors ?? [];
  const online = sensors.filter((s) => s.online).length;

  return (
    <div className="flex-col gap-3">
      <div className="grid" style={{ gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
        <MetricTile label="Online" value={`${online}/${sensors.length || 8}`} tone={online === sensors.length ? "ok" : "warn"} />
        <MetricTile label="Sample Rate" value={((status?.network.sample_rate_hz ?? 16000) / 1000).toFixed(0)} unit="kHz" />
        <MetricTile label="Packets Received" value={status?.network.packets_received ?? 0} />
        <MetricTile label="Packets Dropped" value={status?.network.packets_dropped ?? 0} tone={status?.network.packets_dropped ? "warn" : "ok"} />
      </div>
      <Panel title="Hydrophone Array">
        <ArrayGrid sensors={sensors} />
      </Panel>
    </div>
  );
}
