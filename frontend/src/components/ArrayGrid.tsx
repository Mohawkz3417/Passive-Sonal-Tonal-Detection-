import React from "react";
import type { SensorStatus } from "../types";
import { StatusPill } from "./common";

export function ArrayGrid({ sensors }: { sensors: SensorStatus[] }) {
  if (sensors.length === 0) {
    return <div className="text-tertiary" style={{ fontSize: 12.5 }}>No sensor data available.</div>;
  }
  return (
    <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: 10 }}>
      {sensors.map((s) => (
        <div key={s.id} className="panel fade-in" style={{ padding: "12px 14px" }}>
          <div className="flex items-center justify-between" style={{ marginBottom: 8 }}>
            <span className="mono" style={{ fontWeight: 700, fontSize: 13 }}>{s.label}</span>
            <StatusPill tone={s.online ? "ok" : "off"} pulse={s.online}>{s.online ? "ONLINE" : "OFFLINE"}</StatusPill>
          </div>
          <div style={{ height: 6, background: "var(--bg-inset)", borderRadius: 3, overflow: "hidden", marginBottom: 6 }}>
            <div style={{
              height: "100%", width: `${Math.round((s.signal_level ?? 0) * 100)}%`,
              background: s.clipping ? "var(--status-err)" : "var(--accent)", transition: "width 0.4s ease",
            }} />
          </div>
          <div className="flex justify-between text-tertiary mono" style={{ fontSize: 10.5 }}>
            <span>SIGNAL</span>
            <span>{s.signal_level !== null ? `${(s.signal_level * 100).toFixed(0)}%` : "N/A"}</span>
          </div>
          {s.clipping && <div className="text-err" style={{ fontSize: 10.5, marginTop: 4 }}>⚠ CLIPPING</div>}
          {s.warning && <div className="text-warn" style={{ fontSize: 10.5, marginTop: 4 }}>{s.warning}</div>}
        </div>
      ))}
    </div>
  );
}
