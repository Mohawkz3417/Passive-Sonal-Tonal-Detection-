import React, { useState } from "react";
import { useApp } from "../store";
import { Panel } from "../components/common";
import { toastBus } from "../hooks";

export function SettingsPage() {
  const { settings, updateSettings, theme, toggleTheme, mode, setMode } = useApp();
  const [urls, setUrls] = useState({
    detectionBackendUrl: settings.detectionBackendUrl,
    signalGeneratorUrl: settings.signalGeneratorUrl,
    websocketUrl: settings.websocketUrl,
  });

  const saveUrls = () => {
    updateSettings(urls);
    toastBus.push("Backend configuration saved", "success");
  };

  return (
    <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", gap: 14 }}>
      <Panel title="Appearance">
        <FieldRow label="Theme">
          <div className="flex gap-1">
            <button className="btn" style={{ background: theme === "dark" ? "var(--accent)" : undefined, color: theme === "dark" ? "#061014" : undefined }} onClick={() => theme !== "dark" && toggleTheme()}>Dark</button>
            <button className="btn" style={{ background: theme === "light" ? "var(--accent)" : undefined, color: theme === "light" ? "#061014" : undefined }} onClick={() => theme !== "light" && toggleTheme()}>Light</button>
          </div>
        </FieldRow>
        <FieldRow label="Density">
          <select className="field" value={settings.density} onChange={(e) => updateSettings({ density: e.target.value as "compact" | "comfortable" })}>
            <option value="comfortable">Comfortable</option>
            <option value="compact">Compact</option>
          </select>
        </FieldRow>
      </Panel>

      <Panel title="Data Mode">
        <FieldRow label="Mode">
          <div className="flex gap-1">
            <button className="btn" style={{ background: mode === "demo" ? "var(--accent)" : undefined, color: mode === "demo" ? "#061014" : undefined }} onClick={() => setMode("demo")}>Demo</button>
            <button className="btn" style={{ background: mode === "live" ? "var(--accent)" : undefined, color: mode === "live" ? "#061014" : undefined }} onClick={() => setMode("live")}>Live</button>
          </div>
        </FieldRow>
        <p className="text-tertiary" style={{ fontSize: 11.5 }}>
          Live mode connects to the Detection Backend and WebSocket URLs configured below. If unreachable,
          switch back to Demo to continue reviewing the interface.
        </p>
      </Panel>

      <Panel title="Backend Configuration">
        <FieldRow label="Detection Backend URL">
          <input className="field w-full" value={urls.detectionBackendUrl} onChange={(e) => setUrls((u) => ({ ...u, detectionBackendUrl: e.target.value }))} />
        </FieldRow>
        <FieldRow label="Signal Generator URL">
          <input className="field w-full" value={urls.signalGeneratorUrl} onChange={(e) => setUrls((u) => ({ ...u, signalGeneratorUrl: e.target.value }))} />
        </FieldRow>
        <FieldRow label="WebSocket URL">
          <input className="field w-full" value={urls.websocketUrl} onChange={(e) => setUrls((u) => ({ ...u, websocketUrl: e.target.value }))} />
        </FieldRow>
        <button className="btn primary" onClick={saveUrls}>Save</button>
      </Panel>

      <Panel title="Display">
        <FieldRow label="Refresh Interval">
          <input className="field" type="number" value={settings.refreshIntervalMs} onChange={(e) => updateSettings({ refreshIntervalMs: Number(e.target.value) })} style={{ width: 100 }} /> ms
        </FieldRow>
        <FieldRow label="Show Grid">
          <input type="checkbox" checked={settings.showGrid} onChange={(e) => updateSettings({ showGrid: e.target.checked })} />
        </FieldRow>
        <FieldRow label="Show Event Log">
          <input type="checkbox" checked={settings.showEventLog} onChange={(e) => updateSettings({ showEventLog: e.target.checked })} />
        </FieldRow>
        <FieldRow label="Show Pipeline">
          <input type="checkbox" checked={settings.showPipeline} onChange={(e) => updateSettings({ showPipeline: e.target.checked })} />
        </FieldRow>
      </Panel>
    </div>
  );
}

function FieldRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-2" style={{ padding: "8px 0" }}>
      <span className="text-secondary" style={{ fontSize: 12.5 }}>{label}</span>
      <div className="flex items-center gap-1">{children}</div>
    </div>
  );
}
