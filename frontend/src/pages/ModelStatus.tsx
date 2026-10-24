import React, { useEffect, useMemo, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { useProviderData } from "../hooks";
import { Panel, StatusPill } from "../components/common";
import { useApp } from "../store";
import { DetectionBackendClient } from "../services/restClient";
import type { ModelCompareResult, ModelCompareEntry } from "../types";

export function ModelStatusPage() {
  const { mode, settings } = useApp();
  const { data: status } = useProviderData((p) => p.getStatus());
  const client = useMemo(() => new DetectionBackendClient(settings.detectionBackendUrl), [settings.detectionBackendUrl]);
  const [compare, setCompare] = useState<ModelCompareResult | null>(null);
  const [compareError, setCompareError] = useState<string | null>(null);

  useEffect(() => {
    if (mode !== "live") { setCompare(null); return; }
    let cancelled = false;
    const poll = () => client.modelCompare()
      .then((r) => { if (!cancelled) { setCompare(r); setCompareError(null); } })
      .catch(() => { if (!cancelled) setCompareError("Could not reach /api/model/compare on the Detection Backend."); });
    poll();
    const id = window.setInterval(poll, 4000);
    return () => { cancelled = true; window.clearInterval(id); };
  }, [mode, client]);

  return (
    <div className="flex-col gap-3">
      {mode === "demo" && (
        <Panel title="Model Comparison">
          <div className="text-tertiary" style={{ fontSize: 12.5 }}>
            Model comparison (SegFormer vs U-Net) reads live checkpoints and inference results from
            the Detection Backend's <span className="mono">/api/model/compare</span> — connect a live
            backend to view it. Demo mode shows the primary model summary below only.
          </div>
        </Panel>
      )}
      {mode === "live" && compareError && (
        <Panel title="Model Comparison"><div className="text-err" style={{ fontSize: 12 }}>{compareError}</div></Panel>
      )}
      {mode === "live" && compare && (
        <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", gap: 14 }}>
          <ModelCard title="SegFormer-B0 (primary detector)" entry={compare.segformer} accent="var(--accent)" />
          <ModelCard title="U-Net (baseline / comparison)" entry={compare.unet} accent="var(--status-warn)" />
        </div>
      )}
      {mode === "live" && compare && compare.segformer.available && compare.unet.available && (
        <Panel title="Last Live Detections — Side by Side">
          <div className="text-tertiary" style={{ fontSize: 11, marginBottom: 8 }}>{compare.note}</div>
          <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <DetectionList label="SegFormer" dets={compare.segformer.last_detections} />
            <DetectionList label="U-Net" dets={compare.unet.last_detections} />
          </div>
        </Panel>
      )}

      <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <Panel title="Active Model (runtime status)">
          <Row label="Architecture" value={status?.model.architecture ?? "segformer_b0"} />
          <Row label="Status" value={<StatusPill tone={status?.model.loaded ? "ok" : "off"}>{status?.model.loaded ? "LOADED" : "NOT LOADED"}</StatusPill>} />
          <Row label="Device" value={status?.model.device ?? "cpu"} />
          <Row label="Checkpoint" value={status?.model.checkpoint_path?.split("/").pop() ?? "—"} />
          <Row label="Confidence Threshold" value="0.5" />
          <Row label="Last Inference" value={status?.model.inference_latency_ms ? `${status.model.inference_latency_ms.toFixed(1)} ms` : "—"} />
        </Panel>
        <Panel title="Kalman Tracker">
          <div className="text-secondary" style={{ fontSize: 12.5 }}>
            Multi-target 1D constant-velocity Kalman filter on tonal frequency. A track is reported as
            <b> CONFIRMED</b> only after 3 consecutive matching frames (rejects one-frame false alarms),
            and dropped as <b>LOST</b> after 4 consecutive missed frames. See the <b>Tracking</b> page for
            live per-track history, or <span className="mono">/api/tracks</span>.
          </div>
        </Panel>
      </div>
    </div>
  );
}

function ModelCard({ title, entry, accent }: { title: string; entry: ModelCompareEntry; accent: string }) {
  const history = entry.training_history ?? [];
  return (
    <Panel title={title} actions={<StatusPill tone={entry.available ? "ok" : "off"}>{entry.available ? "LOADED" : "NOT TRAINED"}</StatusPill>}>
      <Row label="Checkpoint" value={entry.checkpoint_path.split("/").pop() ?? "—"} />
      <Row label="Checkpoint File" value={<StatusPill tone={entry.checkpoint_exists ? "ok" : "warn"}>{entry.checkpoint_exists ? "EXISTS" : "MISSING"}</StatusPill>} />
      <Row label="Last Inference" value={entry.last_inference_latency_ms ? `${entry.last_inference_latency_ms.toFixed(1)} ms` : "—"} />
      <Row label="Last Scene SNR" value={entry.last_scene_snr_db !== null && entry.last_scene_snr_db !== undefined ? `${entry.last_scene_snr_db.toFixed(1)} dB` : "—"} />
      {!entry.available && entry.load_error && (
        <div className="text-tertiary" style={{ fontSize: 11, marginTop: 8, whiteSpace: "pre-wrap" }}>{entry.load_error}</div>
      )}
      {history.length > 0 ? (
        <div style={{ marginTop: 12 }}>
          <div style={{ fontSize: 11, color: "var(--text-secondary)", marginBottom: 4 }}>
            Real training history — {history.length} epoch{history.length === 1 ? "" : "s"} (val F1 / loss)
          </div>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={history} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-soft)" />
              <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: "var(--text-tertiary)" }} stroke="var(--border)" />
              <YAxis tick={{ fontSize: 10, fill: "var(--text-tertiary)" }} stroke="var(--border)" width={34} domain={[0, 1]} />
              <Tooltip contentStyle={{ background: "var(--bg-panel-raised)", border: "1px solid var(--border)", borderRadius: 6, fontSize: 11 }} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <Line type="monotone" dataKey="f1" name="val F1" stroke={accent} strokeWidth={2} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="train_loss" name="train loss" stroke="var(--text-tertiary)" strokeWidth={1.5} dot={false} isAnimationActive={false} strokeDasharray="4 3" />
            </LineChart>
          </ResponsiveContainer>
          <div className="flex justify-between" style={{ fontSize: 11, marginTop: 4 }}>
            {(() => {
              // The DEPLOYED checkpoint is the best-F1 epoch across the whole
              // (possibly noisy, resumed-in-rounds) run, not necessarily the
              // LAST epoch trained -- a later epoch can dip below an earlier
              // one on this small CPU-bound training setup. Reporting the max
              // here keeps this label truthful regardless of which epoch the
              // run happened to end on.
              const best = history.reduce((a, b) => (b.f1 > a.f1 ? b : a), history[0]);
              const isLast = best.epoch === history[history.length - 1].epoch;
              return (
                <>
                  <span className="text-tertiary">
                    Deployed (best) val F1: <b className="mono">{best.f1.toFixed(3)}</b>
                    {!isLast && (
                      <span title="A later epoch scored lower -- the best-scoring epoch's weights are what's deployed, not the last epoch trained.">
                        {" "}(epoch {best.epoch}/{history.length})
                      </span>
                    )}
                  </span>
                  <span className="text-tertiary">Deployed (best) val IoU: <b className="mono">{best.iou.toFixed(3)}</b></span>
                </>
              );
            })()}
          </div>
        </div>
      ) : (
        <div className="text-tertiary" style={{ fontSize: 12, marginTop: 10 }}>
          No training history found — this checkpoint has not been trained yet.
        </div>
      )}
    </Panel>
  );
}

function DetectionList({ label, dets }: { label: string; dets: ModelCompareEntry["last_detections"] }) {
  return (
    <div>
      <div style={{ fontSize: 11, color: "var(--text-secondary)", marginBottom: 6 }}>{label} ({dets.length} confirmed track{dets.length === 1 ? "" : "s"})</div>
      {dets.length === 0 ? (
        <div className="text-tertiary" style={{ fontSize: 12 }}>No confirmed detections yet.</div>
      ) : (
        <div className="flex-col gap-1 mono" style={{ fontSize: 12 }}>
          {dets.map((d) => (
            <div key={d.track_id} className="flex justify-between">
              <span>T-{String(d.track_id).padStart(3, "0")} · {d.frequency_hz.toFixed(1)} Hz</span>
              <span className="text-tertiary">conf {d.confidence.toFixed(1)} · {d.track_hits} hits</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between" style={{ padding: "8px 0", borderBottom: "1px solid var(--border-soft)" }}>
      <span className="text-secondary" style={{ fontSize: 12 }}>{label}</span>
      <span className="mono" style={{ fontSize: 12, fontWeight: 600 }}>{value}</span>
    </div>
  );
}
