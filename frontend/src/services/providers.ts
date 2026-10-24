import type {
  SystemStatus, DetectionResult, BeamSpectrumData, SpectrogramData, Track, Contact,
  EventLogEntry, ConnectionState, DemoScenarioId,
} from "../types";
import { DetectionBackendClient } from "./restClient";
import { SonarWebSocket } from "./wsClient";
import { DemoEngine, DEMO_SCENARIOS } from "./demoEngine";
import type { AppSettings } from "../config";

/**
 * The interface every page consumes. Pages import a provider instance
 * from usage sites (see hooks/useProvider.ts) and call these methods --
 * NEVER `fetch`/`WebSocket`/DemoEngine directly. This is what lets
 * switching demo <-> live require zero page rewrites (Part 7/54).
 */
export interface SonarDataProvider {
  readonly kind: "demo" | "live";
  getStatus(): Promise<SystemStatus | null>;
  getLatestDetection(): Promise<DetectionResult | null>;
  getTracks(): Promise<Record<string, Track>>;
  getContacts(): Promise<Contact[]>;
  getBeamSpectrum(): Promise<BeamSpectrumData | null>;
  getSpectrogram(): Promise<SpectrogramData | null>;
  getEvents(): Promise<EventLogEntry[]>;
  getConnectionState(): ConnectionState;
  subscribe(fn: () => void): () => void;
  analyzeFile(kind: "wav" | "npz" | "mp3", file: File): Promise<DetectionResult>;
  // Demo-only controls -- BackendDataProvider implements these as no-ops
  // since a live backend is driven by the real Signal Generator, not by
  // page controls (Part 34 demo drawer).
  demoStart(): void;
  demoPause(): void;
  demoReset(): void;
  demoSetScenario(id: DemoScenarioId): void;
  demoIsRunning(): boolean;
  demoScenario(): DemoScenarioId;
}

// --------------------------------------------------------------------- Demo
export class DemoDataProvider implements SonarDataProvider {
  readonly kind = "demo" as const;
  private engine = new DemoEngine();
  private timer: number | null = null;

  constructor(tickMs = 1000) {
    this.timer = window.setInterval(() => this.engine.tick(), tickMs);
  }
  destroy() {
    if (this.timer) window.clearInterval(this.timer);
  }

  async getStatus() { return this.engine.getSystemStatus(); }
  async getLatestDetection() { return this.engine.getPrimaryDetection(); }
  async getTracks() { return this.engine.getTracks(); }
  async getContacts() { return this.engine.getContacts(); }
  async getBeamSpectrum() { return this.engine.getBeamSpectrum(); }
  async getSpectrogram() { return this.engine.getSpectrogram(); }
  async getEvents() { return this.engine.getEvents(); }
  getConnectionState(): ConnectionState { return this.engine.isRunning() ? "live" : "disconnected"; }
  subscribe(fn: () => void) { return this.engine.subscribe(fn); }
  async analyzeFile(): Promise<DetectionResult> {
    // Simulated analysis result -- clearly a demo value, not a real inference.
    await new Promise((r) => setTimeout(r, 600));
    return this.engine.getPrimaryDetection();
  }
  demoStart() { this.engine.start(); }
  demoPause() { this.engine.pause(); }
  demoReset() { this.engine.reset(); }
  demoSetScenario(id: DemoScenarioId) { this.engine.setScenario(id); }
  demoIsRunning() { return this.engine.isRunning(); }
  demoScenario() { return this.engine.getScenario(); }
}

// --------------------------------------------------------------------- Live backend
export class BackendDataProvider implements SonarDataProvider {
  readonly kind = "live" as const;
  private client: DetectionBackendClient;
  private ws: SonarWebSocket;
  private listeners = new Set<() => void>();
  private connState: ConnectionState = "connecting";

  constructor(baseUrl: string, wsUrl: string) {
    this.client = new DetectionBackendClient(baseUrl);
    this.ws = new SonarWebSocket(wsUrl);
    this.ws.onStateChange((s) => {
      this.connState = s;
      this.listeners.forEach((l) => l());
    });
    this.ws.onMessage(() => this.listeners.forEach((l) => l()));
    this.ws.connect();
  }
  destroy() { this.ws.disconnect(); }

  async getStatus() {
    try { return await this.client.status(); } catch { return null; }
  }
  async getLatestDetection() {
    try {
      const d = await this.client.latestDetection();
      return "detected" in d && "timestamp" in d ? (d as DetectionResult) : null;
    } catch { return null; }
  }
  async getTracks() {
    try { return await this.client.tracks(); } catch { return {}; }
  }
  async getContacts(): Promise<Contact[]> {
    // The backend has no dedicated /api/contacts endpoint -- Contact is a
    // frontend-level view built from Track + the latest DetectionResult
    // (documented adapter, Part 45), not an invented backend concept.
    try {
      const [tracks, latest] = await Promise.all([this.getTracks(), this.getLatestDetection()]);
      return Object.values(tracks).map((t) => {
        const isPrimary = latest?.track_id === t.track_id;
        const lastPoint = t.points[t.points.length - 1];
        return {
          id: `T-${String(t.track_id).padStart(3, "0")}`,
          track_id: t.track_id, status: t.status,
          bearing_deg: lastPoint?.bearing_deg ?? null, frequency_hz: lastPoint?.frequency_hz ?? null,
          snr_db: isPrimary ? latest?.measured_snr_db ?? null : null,
          classification: isPrimary ? latest?.classification ?? null : null,
          confidence: lastPoint?.confidence ?? null,
          first_detected: t.points[0]?.timestamp ?? "", last_updated: lastPoint?.timestamp ?? "",
          bandwidth_hz: isPrimary ? latest?.bandwidth_hz ?? null : null,
          n_harmonics: isPrimary ? latest?.n_harmonics ?? null : null,
          track_hits: t.points.length, track_misses: 0,
        };
      });
    } catch { return []; }
  }
  async getBeamSpectrum() {
    try { const b = await this.client.beamSpectrum(); return "timestamp" in b ? b as BeamSpectrumData : null; }
    catch { return null; }
  }
  async getSpectrogram() {
    try { return await this.client.spectrogram(); } catch { return null; }
  }
  async getEvents(): Promise<EventLogEntry[]> {
    // No backend event-log endpoint exists yet -- documented gap (Part 41),
    // not fabricated. Returns empty rather than inventing log lines.
    return [];
  }
  getConnectionState() { return this.connState; }
  subscribe(fn: () => void) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  async analyzeFile(kind: "wav" | "npz" | "mp3", file: File) {
    if (kind === "wav") return this.client.analyzeWav(file);
    if (kind === "npz") return this.client.analyzeNpz(file);
    return this.client.analyzeMp3(file);
  }
  // Live mode has no local demo controls -- the real scene is driven by the
  // actual Signal Generator app, not by the frontend.
  demoStart() {}
  demoPause() {}
  demoReset() {}
  demoSetScenario() {}
  demoIsRunning() { return false; }
  demoScenario(): DemoScenarioId { return "ship_contact"; }
}

export { DEMO_SCENARIOS };

export function createProvider(mode: "demo" | "live", settings: AppSettings): SonarDataProvider {
  if (mode === "live") return new BackendDataProvider(settings.detectionBackendUrl, settings.websocketUrl);
  return new DemoDataProvider(1000);
}
