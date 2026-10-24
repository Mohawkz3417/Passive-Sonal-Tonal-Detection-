import React, { useState } from "react";
import { useApp } from "../store";
import type { DemoScenarioId } from "../types";
import { Play, Pause, RotateCcw, ChevronDown, ChevronUp } from "lucide-react";
import { toastBus } from "../hooks";

const PRESETS: { id: DemoScenarioId; label: string }[] = [
  { id: "ship_contact", label: "Ship Contact" },
  { id: "low_snr", label: "Low SNR" },
  { id: "biological", label: "Biological" },
  { id: "search", label: "Search" },
];

export function DemoControlDrawer() {
  const { provider } = useApp();
  const [open, setOpen] = useState(true);
  const [, forceRender] = useState(0);
  const running = provider.demoIsRunning();
  const scenario = provider.demoScenario();

  const refresh = () => forceRender((n) => n + 1);

  return (
    <div className="panel fade-in" style={{ padding: 0 }}>
      <div className="flex items-center justify-between" style={{ padding: "10px 16px", cursor: "pointer" }} onClick={() => setOpen((o) => !o)}>
        <span className="panel-title">Demo Control</span>
        {open ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
      </div>
      {open && (
        <div style={{ padding: "0 16px 14px" }} className="flex items-center gap-2" >
          <div className="flex gap-1">
            {PRESETS.map((p) => (
              <button key={p.id} className="btn" style={{
                background: scenario === p.id ? "var(--accent)" : undefined,
                color: scenario === p.id ? "#061014" : undefined,
                borderColor: scenario === p.id ? "var(--accent)" : undefined,
              }} onClick={() => { provider.demoSetScenario(p.id); toastBus.push(`Scenario: ${p.label}`, "info"); refresh(); }}>
                {p.label}
              </button>
            ))}
          </div>
          <div style={{ flex: 1 }} />
          <button className="btn primary" onClick={() => { running ? provider.demoPause() : provider.demoStart(); toastBus.push(running ? "Demo paused" : "Demo started", "success"); refresh(); }}>
            {running ? <Pause size={13} /> : <Play size={13} />} {running ? "Pause" : "Start"}
          </button>
          <button className="btn" onClick={() => { provider.demoReset(); toastBus.push("Demo reset", "info"); refresh(); }}>
            <RotateCcw size={13} /> Reset
          </button>
        </div>
      )}
    </div>
  );
}
