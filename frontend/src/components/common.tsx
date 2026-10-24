import React from "react";
import { useToasts } from "../hooks";
import { CheckCircle2, AlertTriangle, XCircle, Info, X } from "lucide-react";

export type PillTone = "ok" | "warn" | "err" | "off";

export function StatusPill({ tone, children, pulse = false }: { tone: PillTone; children: React.ReactNode; pulse?: boolean }) {
  return (
    <span className={`pill pill-${tone} ${pulse ? "pulse" : ""}`}>
      <span className="pill-dot" />
      {children}
    </span>
  );
}

export function pipelineStateTone(state: string): PillTone {
  if (state === "READY" || state === "RUNNING") return "ok";
  if (state === "WARNING") return "warn";
  if (state === "ERROR") return "err";
  return "off";
}

export function Panel({
  title, actions, children, className = "",
}: { title?: string; actions?: React.ReactNode; children: React.ReactNode; className?: string }) {
  return (
    <div className={`panel fade-in ${className}`}>
      {title && (
        <div className="panel-header">
          <span className="panel-title">{title}</span>
          {actions}
        </div>
      )}
      <div className="panel-body">{children}</div>
    </div>
  );
}

export function EmptyState({ icon, title, message, action }: { icon?: React.ReactNode; title: string; message?: string; action?: React.ReactNode }) {
  return (
    <div className="flex-col items-center gap-2" style={{ padding: "40px 20px", textAlign: "center", color: "var(--text-secondary)" }}>
      {icon}
      <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text-primary)" }}>{title}</div>
      {message && <div style={{ fontSize: 12.5, maxWidth: 360 }}>{message}</div>}
      {action}
    </div>
  );
}

export function Skeleton({ height = 16, width = "100%" }: { height?: number; width?: number | string }) {
  return <div className="skeleton" style={{ height, width }} />;
}

const TOAST_ICON: Record<string, React.ReactNode> = {
  info: <Info size={16} />, success: <CheckCircle2 size={16} />,
  warning: <AlertTriangle size={16} />, error: <XCircle size={16} />,
};
const TOAST_COLOR: Record<string, string> = {
  info: "var(--accent)", success: "var(--status-ok)", warning: "var(--status-warn)", error: "var(--status-err)",
};

export function ToastContainer() {
  const { toasts, dismiss } = useToasts();
  return (
    <div style={{ position: "fixed", bottom: 20, right: 20, display: "flex", flexDirection: "column", gap: 8, zIndex: 1000 }}>
      {toasts.map((t) => (
        <div key={t.id} className="panel scale-in" style={{
          display: "flex", alignItems: "center", gap: 10, padding: "10px 14px",
          minWidth: 260, borderLeft: `3px solid ${TOAST_COLOR[t.tone]}`,
        }}>
          <span style={{ color: TOAST_COLOR[t.tone] }}>{TOAST_ICON[t.tone]}</span>
          <span style={{ fontSize: 12.5, flex: 1 }}>{t.message}</span>
          <button onClick={() => dismiss(t.id)} style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-tertiary)" }}>
            <X size={14} />
          </button>
        </div>
      ))}
    </div>
  );
}

export function MetricTile({ label, value, unit, tone }: { label: string; value: React.ReactNode; unit?: string; tone?: PillTone }) {
  const color = tone === "ok" ? "var(--status-ok)" : tone === "warn" ? "var(--status-warn)" : tone === "err" ? "var(--status-err)" : "var(--text-primary)";
  return (
    <div style={{ padding: "10px 14px", background: "var(--bg-inset)", borderRadius: "var(--radius-md)", border: "1px solid var(--border-soft)" }}>
      <div style={{ fontSize: 10, letterSpacing: "0.07em", color: "var(--text-tertiary)", textTransform: "uppercase", marginBottom: 4 }}>{label}</div>
      <div className="mono" style={{ fontSize: 20, fontWeight: 700, color }}>
        {value} {unit && <span style={{ fontSize: 12, color: "var(--text-secondary)", fontWeight: 500 }}>{unit}</span>}
      </div>
    </div>
  );
}

export function formatTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleTimeString(undefined, { hour12: false });
  } catch { return "—"; }
}

export function formatAgo(iso: string | null | undefined): string {
  if (!iso) return "—";
  const ms = Date.now() - new Date(iso).getTime();
  if (ms < 0 || Number.isNaN(ms)) return "—";
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  return `${Math.floor(m / 60)}h ago`;
}
