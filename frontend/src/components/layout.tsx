import React, { useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import {
  LayoutDashboard, Radar, Users, Route, AudioWaveform, GitBranch, FlaskConical,
  Grid3x3, HeartPulse, Settings as SettingsIcon, Moon, Sun, LogOut, Maximize2, Minimize2,
  ChevronsLeft, ChevronsRight, Cpu, Upload, RadioTower,
} from "lucide-react";
import { useApp } from "../store";
import { useClock, useProviderData } from "../hooks";
import { APP_NAME, APP_SUBTITLE } from "../config";
import { StatusPill } from "./common";

const NAV_SECTIONS: { label: string; items: { to: string; label: string; icon: React.ReactNode; key: string }[] }[] = [
  { label: "Command", items: [{ to: "/dashboard", label: "Dashboard", icon: <LayoutDashboard size={17} />, key: "d" }] },
  {
    label: "Monitoring", items: [
      { to: "/live", label: "Live Sonar", icon: <Radar size={17} />, key: "s" },
      { to: "/contacts", label: "Contacts", icon: <Users size={17} />, key: "c" },
      { to: "/tracking", label: "Tracking", icon: <Route size={17} />, key: "t" },
    ],
  },
  {
    label: "Analysis", items: [
      { to: "/spectrogram", label: "Spectrogram", icon: <AudioWaveform size={17} />, key: "" },
      { to: "/beamforming", label: "Beamforming", icon: <GitBranch size={17} />, key: "" },
      { to: "/acoustic-analysis", label: "Acoustic Analysis", icon: <FlaskConical size={17} />, key: "a" },
      { to: "/input-analysis", label: "Input Analysis", icon: <Upload size={17} />, key: "" },
      { to: "/generator", label: "Signal Generator", icon: <RadioTower size={17} />, key: "" },
    ],
  },
  {
    label: "System", items: [
      { to: "/array", label: "Array Status", icon: <Grid3x3 size={17} />, key: "" },
      { to: "/model", label: "Model Status", icon: <Cpu size={17} />, key: "" },
      { to: "/health", label: "System Health", icon: <HeartPulse size={17} />, key: "h" },
      { to: "/settings", label: "Settings", icon: <SettingsIcon size={17} />, key: "" },
    ],
  },
];

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  return (
    <aside style={{
      width: collapsed ? "var(--sidebar-w-collapsed)" : "var(--sidebar-w)", flexShrink: 0,
      background: "var(--bg-panel)", borderRight: "1px solid var(--border)",
      display: "flex", flexDirection: "column", transition: "width 0.2s ease", overflow: "hidden",
    }}>
      <div style={{ padding: "16px 14px", borderBottom: "1px solid var(--border-soft)", display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{
          width: 30, height: 30, borderRadius: 8, background: "var(--accent)", flexShrink: 0,
          display: "flex", alignItems: "center", justifyContent: "center", color: "#061014",
        }}>
          <Radar size={17} className="sonar-sweep" style={{ animationDuration: "6s" }} />
        </div>
        {!collapsed && (
          <div style={{ minWidth: 0 }}>
            <div style={{ fontWeight: 700, fontSize: 13, letterSpacing: "0.03em", whiteSpace: "nowrap" }}>{APP_NAME}</div>
            <div style={{ fontSize: 9.5, color: "var(--text-tertiary)", whiteSpace: "nowrap" }}>{APP_SUBTITLE}</div>
          </div>
        )}
      </div>
      <nav style={{ flex: 1, overflowY: "auto", padding: "10px 8px" }}>
        {NAV_SECTIONS.map((section) => (
          <div key={section.label} style={{ marginBottom: 14 }}>
            {!collapsed && (
              <div style={{ fontSize: 9.5, letterSpacing: "0.09em", color: "var(--text-tertiary)", padding: "4px 10px", textTransform: "uppercase" }}>
                {section.label}
              </div>
            )}
            {section.items.map((item) => (
              <NavLink
                key={item.to} to={item.to} title={collapsed ? item.label : undefined}
                style={({ isActive }) => ({
                  display: "flex", alignItems: "center", gap: 10, padding: "8px 10px", borderRadius: "var(--radius-md)",
                  color: isActive ? "var(--accent)" : "var(--text-secondary)",
                  background: isActive ? "var(--accent-glow)" : "transparent",
                  textDecoration: "none", fontSize: 12.5, fontWeight: 600, marginBottom: 2,
                  whiteSpace: "nowrap", overflow: "hidden",
                })}
              >
                {item.icon}
                {!collapsed && item.label}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>
      <button className="btn" style={{ margin: 10, justifyContent: "center" }} onClick={() => setCollapsed((c) => !c)}>
        {collapsed ? <ChevronsRight size={15} /> : <><ChevronsLeft size={15} /> Collapse</>}
      </button>
    </aside>
  );
}

export function Header() {
  const { mode, setMode, theme, toggleTheme, logout, operatorId, provider } = useApp();
  const clock = useClock();
  const navigate = useNavigate();
  const [fullscreen, setFullscreen] = useState(false);
  const { data: status } = useProviderData((p) => p.getStatus(), 2000);

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen?.();
      setFullscreen(true);
    } else {
      document.exitFullscreen?.();
      setFullscreen(false);
    }
  };

  const connState = provider.getConnectionState();
  const arrayOnline = status?.sensors?.filter((s) => s.online).length ?? 0;
  const arrayTotal = status?.sensors?.length ?? 8;

  return (
    <header style={{
      height: "var(--header-h)", flexShrink: 0, borderBottom: "1px solid var(--border)",
      background: "var(--bg-panel)", display: "flex", alignItems: "center", padding: "0 18px",
      gap: 20, justifyContent: "space-between",
    }}>
      <div className="flex items-center gap-3">
        <StatusPill tone={connState === "live" ? "ok" : mode === "demo" ? "warn" : "off"} pulse={connState === "live"}>
          {mode === "demo" ? "DEMO MODE" : connState === "live" ? "SYSTEM ONLINE" : "DISCONNECTED"}
        </StatusPill>
        <div className="mono text-secondary" style={{ fontSize: 11.5 }}>
          ARRAY <span style={{ color: "var(--text-primary)" }}>{arrayOnline}/{arrayTotal}</span>
        </div>
        <div className="mono text-secondary" style={{ fontSize: 11.5 }}>
          {status?.network?.sample_rate_hz ? (status.network.sample_rate_hz / 1000).toFixed(0) : 16} kHz
        </div>
        <StatusPill tone={status?.network?.connected ? "ok" : "off"}>
          UDP {status?.network?.connected ? "CONNECTED" : "IDLE"}
        </StatusPill>
      </div>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1" style={{ background: "var(--bg-inset)", borderRadius: 999, padding: 3, border: "1px solid var(--border-soft)" }}>
          <button
            className="btn" style={{ padding: "4px 12px", border: "none", background: mode === "demo" ? "var(--accent)" : "transparent", color: mode === "demo" ? "#061014" : "var(--text-secondary)" }}
            onClick={() => setMode("demo")}>DEMO</button>
          <button
            className="btn" style={{ padding: "4px 12px", border: "none", background: mode === "live" ? "var(--accent)" : "transparent", color: mode === "live" ? "#061014" : "var(--text-secondary)" }}
            onClick={() => setMode("live")}>LIVE</button>
        </div>
        <div className="mono text-secondary" style={{ fontSize: 12 }}>
          {clock.toISOString().slice(11, 19)} UTC
        </div>
        <button className="btn" onClick={toggleTheme} title="Toggle theme">
          {theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
        </button>
        <button className="btn" onClick={toggleFullscreen} title="Fullscreen (F)">
          {fullscreen ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
        </button>
        <div className="flex items-center gap-2" style={{ paddingLeft: 10, borderLeft: "1px solid var(--border)" }}>
          <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>{operatorId}</span>
          <button className="btn" onClick={() => { logout(); navigate("/login"); }} title="Log out">
            <LogOut size={14} />
          </button>
        </div>
      </div>
    </header>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="app-shell">
      <Sidebar />
      <div className="app-main">
        <Header />
        <main className="app-content page-transition">{children}</main>
      </div>
    </div>
  );
}
