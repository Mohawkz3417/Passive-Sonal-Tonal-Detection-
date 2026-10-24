/**
 * Single source of truth for backend URLs and default settings.
 * Values match the actual backend apps as configured in this project:
 *   detection-backend-app  -> uvicorn ... --port 8000  (config.yaml: api.detection_port)
 *   signal-generator-app   -> uvicorn ... --port 8001  (config.yaml: api.generator_port)
 *   WebSocket              -> ws://<detection host>:<detection port>/ws/live
 * No other file in this app should hardcode a URL -- read from here, or
 * from useSettings() (Settings page persists overrides to localStorage).
 */

const LS_KEY = "sonar-command:settings";

export interface AppSettings {
  detectionBackendUrl: string;
  signalGeneratorUrl: string;
  websocketUrl: string;
  defaultMode: "demo" | "live";
  theme: "dark" | "light";
  density: "compact" | "comfortable";
  refreshIntervalMs: number;
  sweepSpeedMs: number;
  showGrid: boolean;
  showEventLog: boolean;
  showPipeline: boolean;
}

export const DEFAULT_SETTINGS: AppSettings = {
  detectionBackendUrl: "http://127.0.0.1:8000",
  signalGeneratorUrl: "http://127.0.0.1:8001",
  websocketUrl: "ws://127.0.0.1:8000/ws/live",
  defaultMode: "demo",
  theme: "dark",
  density: "comfortable",
  refreshIntervalMs: 1500,
  sweepSpeedMs: 4000,
  showGrid: true,
  showEventLog: true,
  showPipeline: true,
};

export function loadSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return { ...DEFAULT_SETTINGS };
    return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function saveSettings(settings: AppSettings) {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(settings));
  } catch {
    /* ignore quota errors -- non-critical */
  }
}

export const APP_NAME = "SONAR COMMAND";
export const APP_SUBTITLE = "Intelligent Passive Sonar";
export const APP_TAGLINE = "Acoustic Monitoring & Tonal Detection System";
export const APP_VERSION = "v1.0.0";

// Demo-only local auth -- explicitly NOT production security (Part 48).
export const DEMO_OPERATOR_ID = "operator";
export const DEMO_PASSWORD = "sonar123";
