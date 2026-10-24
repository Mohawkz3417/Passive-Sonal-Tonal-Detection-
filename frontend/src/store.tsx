import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { DEFAULT_SETTINGS, DEMO_OPERATOR_ID, DEMO_PASSWORD, loadSettings, saveSettings, type AppSettings } from "./config";
import { createProvider, type SonarDataProvider } from "./services/providers";
import type { DataMode } from "./types";

const AUTH_KEY = "sonar-command:auth";

interface AppState {
  settings: AppSettings;
  updateSettings: (patch: Partial<AppSettings>) => void;
  mode: DataMode;
  setMode: (m: DataMode) => void;
  provider: SonarDataProvider;
  theme: "dark" | "light";
  toggleTheme: () => void;
  isAuthed: boolean;
  login: (operatorId: string, password: string) => boolean;
  logout: () => void;
  operatorId: string | null;
}

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [settings, setSettings] = useState<AppSettings>(() => loadSettings());
  const [mode, setModeState] = useState<DataMode>(settings.defaultMode);
  const [theme, setTheme] = useState<"dark" | "light">(settings.theme);
  const [operatorId, setOperatorId] = useState<string | null>(() => sessionStorage.getItem(AUTH_KEY));
  const [provider, setProvider] = useState<SonarDataProvider>(() => createProvider(mode, settings));

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  useEffect(() => {
    // Recreate the provider whenever mode or backend URLs change -- pages
    // never construct providers themselves (Part 54).
    const next = createProvider(mode, settings);
    setProvider((prev) => {
      (prev as unknown as { destroy?: () => void }).destroy?.();
      return next;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, settings.detectionBackendUrl, settings.websocketUrl]);

  const updateSettings = useCallback((patch: Partial<AppSettings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      saveSettings(next);
      return next;
    });
    if (patch.theme) setTheme(patch.theme);
  }, []);

  const setMode = useCallback((m: DataMode) => {
    setModeState(m);
    updateSettings({ defaultMode: m });
  }, [updateSettings]);

  const toggleTheme = useCallback(() => {
    setTheme((t) => {
      const next = t === "dark" ? "light" : "dark";
      updateSettings({ theme: next });
      return next;
    });
  }, [updateSettings]);

  const login = useCallback((id: string, password: string) => {
    if (id === DEMO_OPERATOR_ID && password === DEMO_PASSWORD) {
      sessionStorage.setItem(AUTH_KEY, id);
      setOperatorId(id);
      return true;
    }
    return false;
  }, []);

  const logout = useCallback(() => {
    sessionStorage.removeItem(AUTH_KEY);
    setOperatorId(null);
  }, []);

  const value = useMemo<AppState>(() => ({
    settings, updateSettings, mode, setMode, provider, theme, toggleTheme,
    isAuthed: !!operatorId, login, logout, operatorId,
  }), [settings, updateSettings, mode, setMode, provider, theme, toggleTheme, operatorId, login, logout]);

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp(): AppState {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used within AppProvider");
  return ctx;
}

export { DEFAULT_SETTINGS };
