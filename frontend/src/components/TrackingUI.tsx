import React from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import type { Track } from "../types";
import { EmptyState, StatusPill } from "./common";
import { Route } from "lucide-react";

export function TrackHistoryChart({ track, field, label, color, unit }: {
  track: Track; field: "bearing_deg" | "frequency_hz" | "confidence"; label: string; color: string; unit: string;
}) {
  const data = track.points.map((p, i) => ({ i, v: p[field] }));
  if (data.length < 2) {
    return <div className="text-tertiary" style={{ fontSize: 12, padding: 20 }}>Insufficient history yet.</div>;
  }
  return (
    <div>
      <div style={{ fontSize: 11, color: "var(--text-secondary)", marginBottom: 4 }}>{label}</div>
      <ResponsiveContainer width="100%" height={140}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: -22, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border-soft)" />
          <XAxis dataKey="i" hide />
          <YAxis tick={{ fontSize: 10, fill: "var(--text-tertiary)" }} stroke="var(--border)" width={40} />
          <Tooltip contentStyle={{ background: "var(--bg-panel-raised)", border: "1px solid var(--border)", borderRadius: 6, fontSize: 11 }}
                   formatter={(v: number) => [`${v.toFixed(2)} ${unit}`, label]} labelFormatter={() => ""} />
          <Line type="monotone" dataKey="v" stroke={color} strokeWidth={2} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function TrackList({ tracks, selectedId, onSelect }: { tracks: Track[]; selectedId: number | null; onSelect: (id: number) => void }) {
  if (tracks.length === 0) return <EmptyState icon={<Route size={28} />} title="No tracks" message="Tracks appear once a contact is confirmed across multiple frames." />;
  return (
    <div className="flex-col gap-1">
      {tracks.map((t) => {
        const last = t.points[t.points.length - 1];
        return (
          <div key={t.track_id} onClick={() => onSelect(t.track_id)}
               className="flex items-center justify-between"
               style={{
                 padding: "10px 12px", borderRadius: 6, cursor: "pointer",
                 background: t.track_id === selectedId ? "var(--accent-glow)" : "var(--bg-inset)",
                 border: "1px solid " + (t.track_id === selectedId ? "var(--accent)" : "var(--border-soft)"),
               }}>
            <span className="mono" style={{ fontWeight: 700 }}>T-{String(t.track_id).padStart(3, "0")}</span>
            <StatusPill tone={t.status === "CONFIRMED" ? "ok" : "warn"}>{t.status}</StatusPill>
            <span className="mono text-secondary" style={{ fontSize: 11 }}>{last?.frequency_hz?.toFixed(0) ?? "—"} Hz</span>
            <span className="mono text-tertiary" style={{ fontSize: 11 }}>{t.points.length} pts</span>
          </div>
        );
      })}
    </div>
  );
}
