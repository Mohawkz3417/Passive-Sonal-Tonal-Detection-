import React, { useEffect, useState } from "react";
import type { Contact } from "../types";

/**
 * Bearing/contact view -- deliberately NOT presented as a calibrated range
 * PPI (Part 13): this system produces bearing (DOA), not range. Rings are
 * unlabeled reference circles, not distance markers.
 */
export function PPIDisplay({
  contacts, selectedId, onSelect, sweeping = true,
}: { contacts: Contact[]; selectedId?: string | null; onSelect?: (id: string) => void; sweeping?: boolean }) {
  const size = 360;
  const cx = size / 2;
  const cy = size / 2;
  const r = size / 2 - 34;
  const [sweepAngle, setSweepAngle] = useState(-90);

  useEffect(() => {
    if (!sweeping) return;
    let raf: number;
    const start = performance.now();
    const loop = (t: number) => {
      const elapsed = ((t - start) / 4000) % 1;
      setSweepAngle(-90 + elapsed * 180);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [sweeping]);

  // bearing (-90..+90) -> point on the semicircle, 0deg = straight up
  const toXY = (bearingDeg: number, radius: number) => {
    const rad = ((bearingDeg - 90) * Math.PI) / 180;
    return { x: cx + radius * Math.cos(rad), y: cy + radius * Math.sin(rad) };
  };

  const rings = [0.33, 0.66, 1.0];
  const ticks = [-90, -60, -30, 0, 30, 60, 90];

  return (
    <div style={{ position: "relative", width: size, height: size + 20, margin: "0 auto" }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <defs>
          <radialGradient id="ppi-bg" cx="50%" cy="100%" r="100%">
            <stop offset="0%" stopColor="var(--bg-inset)" />
            <stop offset="100%" stopColor="var(--bg-app)" />
          </radialGradient>
        </defs>
        <rect x={0} y={0} width={size} height={size} fill="url(#ppi-bg)" rx={12} />

        {/* range-reference rings (unlabeled -- not calibrated distance) */}
        {rings.map((f) => (
          <path key={f} d={describeArc(cx, cy, r * f, -90, 90)} fill="none" stroke="var(--border)" strokeWidth={1} />
        ))}
        {/* radial spokes at bearing ticks */}
        {ticks.map((t) => {
          const p = toXY(t, r);
          return <line key={t} x1={cx} y1={cy} x2={p.x} y2={p.y} stroke="var(--border-soft)" strokeWidth={1} />;
        })}
        {/* baseline */}
        <line x1={cx - r} y1={cy} x2={cx + r} y2={cy} stroke="var(--border)" strokeWidth={1} />

        {/* sweep */}
        {sweeping && (
          <line
            x1={cx} y1={cy} x2={toXY(sweepAngle, r).x} y2={toXY(sweepAngle, r).y}
            stroke="var(--accent)" strokeWidth={2} strokeLinecap="round" opacity={0.85}
          />
        )}
        {sweeping && (
          <path d={describeArc(cx, cy, r, sweepAngle - 18, sweepAngle)} fill="none" stroke="var(--accent)" strokeWidth={14} opacity={0.08} />
        )}

        {/* bearing tick labels */}
        {ticks.map((t) => {
          const p = toXY(t, r + 16);
          return (
            <text key={t} x={p.x} y={p.y} fontSize={9.5} fill="var(--text-tertiary)" textAnchor="middle" dominantBaseline="middle" fontFamily="var(--font-mono)">
              {t}°
            </text>
          );
        })}

        {/* contacts */}
        {contacts.map((c) => {
          if (c.bearing_deg === null) return null;
          const p = toXY(c.bearing_deg, r * 0.86);
          const isSel = c.id === selectedId;
          const color = c.status === "CONFIRMED" ? "var(--status-ok)" : "var(--status-warn)";
          return (
            <g key={c.id} onClick={() => onSelect?.(c.id)} style={{ cursor: "pointer" }} className="scale-in">
              {isSel && <circle cx={p.x} cy={p.y} r={13} fill="none" stroke={color} strokeWidth={1.5} opacity={0.6}>
                <animate attributeName="r" values="10;16;10" dur="1.6s" repeatCount="indefinite" />
                <animate attributeName="opacity" values="0.7;0.1;0.7" dur="1.6s" repeatCount="indefinite" />
              </circle>}
              <circle cx={p.x} cy={p.y} r={5} fill={color} stroke="var(--bg-app)" strokeWidth={1.5} />
              <text x={p.x} y={p.y - 12} fontSize={9.5} fontWeight={700} fill={isSel ? color : "var(--text-secondary)"} textAnchor="middle" fontFamily="var(--font-mono)">
                {c.id}
              </text>
            </g>
          );
        })}
      </svg>
      <div style={{ textAlign: "center", fontSize: 9.5, letterSpacing: "0.08em", color: "var(--text-tertiary)", marginTop: 2, textTransform: "uppercase" }}>
        Bearing / Contact View — not calibrated range
      </div>
    </div>
  );
}

function polarToCartesian(cx: number, cy: number, r: number, angleDeg: number) {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}
function describeArc(cx: number, cy: number, r: number, startAngle: number, endAngle: number) {
  const start = polarToCartesian(cx, cy, r, endAngle);
  const end = polarToCartesian(cx, cy, r, startAngle);
  const largeArc = endAngle - startAngle <= 180 ? 0 : 1;
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 0 ${end.x} ${end.y}`;
}
