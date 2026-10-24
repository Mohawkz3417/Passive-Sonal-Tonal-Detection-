import React, { useState } from "react";
import type { EventLogEntry } from "../types";
import { StatusPill, pipelineStateTone, EmptyState } from "./common";
import { Terminal, Pause, Play } from "lucide-react";

const STAGE_LABELS: Record<string, string> = {
  INPUT: "Input", ARRAY_PROCESSING: "Array", BEAMFORMER: "Beamform", DOA: "DOA",
  SEGFORMER: "SegFormer", POSTPROCESSING: "CFAR/Post", CLASSIFIER: "Classifier",
  TRACKER: "Tracker", ETHERNET: "Ethernet",
};

export function PipelineStages({ pipeline }: { pipeline: Record<string, string> | undefined }) {
  const order = ["INPUT", "ARRAY_PROCESSING", "BEAMFORMER", "DOA", "SEGFORMER", "POSTPROCESSING", "CLASSIFIER", "TRACKER"];
  return (
    <div className="flex items-center" style={{ overflowX: "auto", paddingBottom: 4 }}>
      {order.map((key, i) => {
        const state = pipeline?.[key] ?? "UNAVAILABLE";
        return (
          <React.Fragment key={key}>
            <div className="flex-col items-center gap-1" style={{ minWidth: 78 }}>
              <StatusPill tone={pipelineStateTone(state)} pulse={state === "RUNNING"}>
                {STAGE_LABELS[key]}
              </StatusPill>
            </div>
            {i < order.length - 1 && <div style={{ width: 16, height: 1, background: "var(--border)", flexShrink: 0 }} />}
          </React.Fragment>
        );
      })}
    </div>
  );
}

export function EventLog({ events }: { events: EventLogEntry[] }) {
  const [paused, setPaused] = useState(false);
  const [filter, setFilter] = useState<string>("all");
  const [frozen, setFrozen] = useState<EventLogEntry[]>([]);

  const shown = paused ? frozen : events;
  const filtered = filter === "all" ? shown : shown.filter((e) => e.severity === filter);

  return (
    <div>
      <div className="flex items-center justify-between gap-2" style={{ marginBottom: 8 }}>
        <div className="flex items-center gap-1">
          <select className="field" value={filter} onChange={(e) => setFilter(e.target.value)} style={{ fontSize: 11 }}>
            <option value="all">All</option>
            <option value="info">Info</option>
            <option value="warning">Warning</option>
            <option value="error">Error</option>
          </select>
        </div>
        <div className="flex gap-1">
          <button className="btn" onClick={() => { if (!paused) setFrozen(events); setPaused((p) => !p); }}>
            {paused ? <Play size={13} /> : <Pause size={13} />} {paused ? "Resume" : "Pause"}
          </button>
        </div>
      </div>
      <div className="mono" style={{ maxHeight: 260, overflowY: "auto", fontSize: 11.5, background: "var(--bg-inset)", borderRadius: 6, padding: 10 }}>
        {filtered.length === 0 ? (
          <EmptyState icon={<Terminal size={22} />} title="No events yet" />
        ) : filtered.map((e) => (
          <div key={e.id} className="flex gap-2 fade-in" style={{ padding: "3px 0", borderBottom: "1px solid var(--border-soft)" }}>
            <span className="text-tertiary">{new Date(e.timestamp).toLocaleTimeString(undefined, { hour12: false })}</span>
            <span style={{
              color: e.severity === "error" ? "var(--status-err)" : e.severity === "warning" ? "var(--status-warn)" : "var(--accent)",
              fontWeight: 700, minWidth: 52,
            }}>{e.severity.toUpperCase()}</span>
            <span style={{ color: "var(--text-primary)" }}>{e.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
