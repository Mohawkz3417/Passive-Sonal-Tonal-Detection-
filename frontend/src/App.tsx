import React, { useEffect } from "react";
import { BrowserRouter, Routes, Route, Navigate, useNavigate, useLocation } from "react-router-dom";
import { AppProvider, useApp } from "./store";
import { AppShell } from "./components/layout";
import { ToastContainer } from "./components/common";
import { Login } from "./pages/Login";
import { Dashboard } from "./pages/Dashboard";
import { LiveSonar } from "./pages/LiveSonar";
import { Contacts } from "./pages/Contacts";
import { Tracking } from "./pages/Tracking";
import { SpectrogramPage } from "./pages/Spectrogram";
import { BeamformingPage } from "./pages/Beamforming";
import { AcousticAnalysisPage } from "./pages/AcousticAnalysis";
import { ArrayStatusPage } from "./pages/ArrayStatus";
import { ModelStatusPage } from "./pages/ModelStatus";
import { SystemHealthPage } from "./pages/SystemHealth";
import { SettingsPage } from "./pages/Settings";
import { InputAnalysisPage } from "./pages/InputAnalysis";
import { SignalGeneratorPage } from "./pages/SignalGeneratorPage";

function RequireAuth({ children }: { children: React.JSX.Element }) {
  const { isAuthed } = useApp();
  const location = useLocation();
  if (!isAuthed) return <Navigate to="/login" state={{ from: location }} replace />;
  return children;
}

const SHORTCUT_ROUTES: Record<string, string> = {
  d: "/dashboard", s: "/live", c: "/contacts", t: "/tracking", a: "/acoustic-analysis", h: "/health",
};

function KeyboardShortcuts() {
  const navigate = useNavigate();
  const { isAuthed } = useApp();
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (!isAuthed) return;
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.key === "f" || e.key === "F") {
        if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
        else document.exitFullscreen?.();
        return;
      }
      const route = SHORTCUT_ROUTES[e.key.toLowerCase()];
      if (route) navigate(route);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [navigate, isAuthed]);
  return null;
}

function AppRoutes() {
  return (
    <>
      <KeyboardShortcuts />
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/*" element={
          <RequireAuth>
            <AppShell>
              <Routes>
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/live" element={<LiveSonar />} />
                <Route path="/contacts" element={<Contacts />} />
                <Route path="/tracking" element={<Tracking />} />
                <Route path="/spectrogram" element={<SpectrogramPage />} />
                <Route path="/beamforming" element={<BeamformingPage />} />
                <Route path="/acoustic-analysis" element={<AcousticAnalysisPage />} />
                <Route path="/input-analysis" element={<InputAnalysisPage />} />
                <Route path="/generator" element={<SignalGeneratorPage />} />
                <Route path="/array" element={<ArrayStatusPage />} />
                <Route path="/model" element={<ModelStatusPage />} />
                <Route path="/health" element={<SystemHealthPage />} />
                <Route path="/settings" element={<SettingsPage />} />
                <Route path="*" element={<Navigate to="/dashboard" replace />} />
              </Routes>
            </AppShell>
          </RequireAuth>
        } />
      </Routes>
    </>
  );
}

export default function App() {
  return (
    <AppProvider>
      <BrowserRouter>
        <AppRoutes />
        <ToastContainer />
      </BrowserRouter>
    </AppProvider>
  );
}
