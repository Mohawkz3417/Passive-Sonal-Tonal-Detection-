import React, { useState } from "react";
import { useProviderData } from "../hooks";
import { Panel } from "../components/common";
import { TrackList, TrackHistoryChart } from "../components/TrackingUI";

export function Tracking() {
  const { data: tracksMap } = useProviderData((p) => p.getTracks());
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const tracks = Object.values(tracksMap ?? {});
  const selected = tracks.find((t) => t.track_id === selectedId) ?? tracks[0] ?? null;

  return (
    <div className="grid" style={{ gridTemplateColumns: "1fr 1.6fr", gap: 14 }}>
      <Panel title="Tracks">
        <TrackList tracks={tracks} selectedId={selected?.track_id ?? null} onSelect={setSelectedId} />
      </Panel>
      <Panel title={selected ? `Track T-${String(selected.track_id).padStart(3, "0")} History` : "Track History"}>
        {selected ? (
          <div className="flex-col gap-3">
            <TrackHistoryChart track={selected} field="bearing_deg" label="Bearing vs Time" color="var(--accent)" unit="°" />
            <TrackHistoryChart track={selected} field="frequency_hz" label="Frequency vs Time" color="var(--status-ok)" unit="Hz" />
            <TrackHistoryChart track={selected} field="confidence" label="Confidence vs Time" color="var(--status-warn)" unit="" />
          </div>
        ) : <div className="text-tertiary" style={{ fontSize: 12.5 }}>Select a track to view its history.</div>}
      </Panel>
    </div>
  );
}
