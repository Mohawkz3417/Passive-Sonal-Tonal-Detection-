import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useApp } from "../store";
import { APP_NAME, APP_SUBTITLE, APP_TAGLINE, APP_VERSION, DEMO_OPERATOR_ID, DEMO_PASSWORD } from "../config";
import { Radar, Eye, EyeOff } from "lucide-react";
import { toastBus } from "../hooks";

export function Login() {
  const { login } = useApp();
  const navigate = useNavigate();
  const [operatorId, setOperatorId] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [remember, setRemember] = useState(true);
  const [error, setError] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (login(operatorId, password)) {
      toastBus.push("Session started", "success");
      navigate("/dashboard");
    } else {
      setError("Invalid operator ID or password.");
    }
  };

  return (
    <div style={{
      minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center",
      background: "radial-gradient(circle at 50% 20%, var(--bg-panel-raised), var(--bg-app) 70%)", position: "relative", overflow: "hidden",
    }}>
      <BackgroundSweep />
      <form onSubmit={handleSubmit} className="panel scale-in" style={{ width: 380, padding: 34, position: "relative", zIndex: 1 }}>
        <div className="flex-col items-center" style={{ marginBottom: 26 }}>
          <div style={{
            width: 54, height: 54, borderRadius: 14, background: "var(--accent)", display: "flex",
            alignItems: "center", justifyContent: "center", marginBottom: 14, boxShadow: "0 0 24px var(--accent-glow)",
          }}>
            <Radar size={28} color="#061014" className="sonar-sweep" style={{ animationDuration: "5s" }} />
          </div>
          <div style={{ fontSize: 20, fontWeight: 800, letterSpacing: "0.06em" }}>{APP_NAME}</div>
          <div style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 6 }}>{APP_SUBTITLE}</div>
          <div style={{ fontSize: 11, color: "var(--text-tertiary)" }}>{APP_TAGLINE}</div>
        </div>

        <label style={{ fontSize: 11, color: "var(--text-secondary)", display: "block", marginBottom: 5 }}>OPERATOR ID</label>
        <input className="field w-full" style={{ marginBottom: 14 }} value={operatorId}
               onChange={(e) => setOperatorId(e.target.value)} placeholder="operator" autoFocus />

        <label style={{ fontSize: 11, color: "var(--text-secondary)", display: "block", marginBottom: 5 }}>PASSWORD</label>
        <div style={{ position: "relative", marginBottom: 10 }}>
          <input className="field w-full" type={showPw ? "text" : "password"} value={password}
                 onChange={(e) => setPassword(e.target.value)} placeholder="••••••••" />
          <button type="button" onClick={() => setShowPw((v) => !v)}
                  style={{ position: "absolute", right: 8, top: 6, background: "none", border: "none", cursor: "pointer", color: "var(--text-tertiary)" }}>
            {showPw ? <EyeOff size={15} /> : <Eye size={15} />}
          </button>
        </div>

        {error && <div className="text-err" style={{ fontSize: 12, marginBottom: 10 }}>{error}</div>}

        <label className="flex items-center gap-1" style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 18, cursor: "pointer" }}>
          <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} /> Remember session
        </label>

        <button type="submit" className="btn primary w-full" style={{ justifyContent: "center", padding: "10px 0", fontSize: 13 }}>
          SIGN IN
        </button>

        <div className="flex items-center justify-center gap-1" style={{ marginTop: 18 }}>
          <span className="pill pill-ok"><span className="pill-dot" />SYSTEM READY</span>
        </div>
        <div style={{ textAlign: "center", fontSize: 10.5, color: "var(--text-tertiary)", marginTop: 10 }}>
          Demo credentials: <span className="mono">{DEMO_OPERATOR_ID}</span> / <span className="mono">{DEMO_PASSWORD}</span>
        </div>
        <div style={{ textAlign: "center", fontSize: 10, color: "var(--text-tertiary)", marginTop: 14 }}>{APP_VERSION}</div>
      </form>
    </div>
  );
}

function BackgroundSweep() {
  return (
    <svg width="600" height="600" style={{ position: "absolute", opacity: 0.25 }} viewBox="0 0 600 600">
      {[100, 175, 250].map((r) => <circle key={r} cx={300} cy={300} r={r} fill="none" stroke="var(--border)" strokeWidth={1} />)}
      <line x1={300} y1={300} x2={300} y2={50} stroke="var(--accent)" strokeWidth={2} className="sonar-sweep" style={{ transformOrigin: "300px 300px", animationDuration: "6s" }} />
    </svg>
  );
}
