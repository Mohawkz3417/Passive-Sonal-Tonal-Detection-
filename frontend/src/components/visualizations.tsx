import React, { useEffect, useRef } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer } from "recharts";
import type { BeamSpectrumData, SpectrogramData } from "../types";
import { EmptyState } from "./common";
import { Activity } from "lucide-react";

export function BeamChart({ data }: { data: BeamSpectrumData | null }) {
  if (!data || data.angles_deg.length === 0) {
    return <EmptyState icon={<Activity size={28} />} title="No beam data" message="Awaiting array processing output." />;
  }
  const chartData = data.angles_deg.map((a, i) => ({ angle: a, power: data.power_db[i] }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={chartData} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-soft)" />
        <XAxis dataKey="angle" tick={{ fontSize: 10, fill: "var(--text-tertiary)" }} tickFormatter={(v) => `${v}°`}
               stroke="var(--border)" />
        <YAxis tick={{ fontSize: 10, fill: "var(--text-tertiary)" }} stroke="var(--border)" width={44}
               label={{ value: "dB", angle: -90, position: "insideLeft", fontSize: 10, fill: "var(--text-tertiary)" }} />
        <Tooltip
          contentStyle={{ background: "var(--bg-panel-raised)", border: "1px solid var(--border)", borderRadius: 6, fontSize: 11 }}
          labelFormatter={(v) => `${v}°`} formatter={(v: number) => [`${v.toFixed(1)} dB`, "Power"]}
        />
        {data.peak_angle_deg !== null && (
          <ReferenceLine x={data.peak_angle_deg} stroke="var(--status-warn)" strokeDasharray="4 3"
                         label={{ value: `${data.peak_angle_deg.toFixed(0)}°`, fontSize: 10, fill: "var(--status-warn)", position: "top" }} />
        )}
        <Line type="monotone" dataKey="power" stroke="var(--accent)" strokeWidth={2} dot={false} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function Waterfall({ data, height = 260 }: { data: SpectrogramData | null; height?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data?.available || !data.magnitude || !data.frequencies_hz) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const nFreq = data.magnitude.length;
    const nTime = data.magnitude[0]?.length ?? 0;
    if (nFreq === 0 || nTime === 0) return;

    let vmin = Infinity, vmax = -Infinity;
    for (const row of data.magnitude) for (const v of row) { if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
    const cellW = W / nTime, cellH = H / nFreq;

    for (let f = 0; f < nFreq; f++) {
      for (let t = 0; t < nTime; t++) {
        const norm = Math.max(0, Math.min(1, (data.magnitude[f][t] - vmin) / (vmax - vmin + 1e-9)));
        const g = Math.floor(norm * 200) + 20;
        ctx.fillStyle = `rgb(0, ${g}, ${Math.floor(g * 0.55)})`;
        ctx.fillRect(t * cellW, H - (f + 1) * cellH, cellW + 1, cellH + 1);
        if (data.detection_mask?.[f]?.[t]) {
          ctx.fillStyle = "rgba(242, 85, 95, 0.55)";
          ctx.fillRect(t * cellW, H - (f + 1) * cellH, cellW + 1, cellH + 1);
        }
      }
    }
  }, [data]);

  if (!data?.available) {
    return <EmptyState icon={<Activity size={28} />} title="No spectrogram data" message={data?.note ?? "Awaiting acquisition."} />;
  }

  const freqs = data.frequencies_hz ?? [];
  return (
    <div>
      <div style={{ position: "relative" }}>
        <canvas ref={canvasRef} width={900} height={height} style={{ width: "100%", height, borderRadius: 6, display: "block", background: "#020403" }} />
        <div style={{ position: "absolute", left: 6, top: 4, fontSize: 9.5, color: "rgba(255,255,255,0.55)" }} className="mono">
          {freqs[freqs.length - 1]?.toFixed(0)} Hz
        </div>
        <div style={{ position: "absolute", left: 6, bottom: 4, fontSize: 9.5, color: "rgba(255,255,255,0.55)" }} className="mono">
          {freqs[0]?.toFixed(0)} Hz
        </div>
      </div>
      <div className="flex justify-between text-tertiary mono" style={{ fontSize: 9.5, marginTop: 4 }}>
        <span>← time</span>
        <span>tonal detection mask shown in red</span>
      </div>
    </div>
  );
}
