import type {
  SystemStatus, DetectionResult, BeamSpectrumData, SpectrogramData, Track,
  ModelCompareResult, RealDatasetInfo,
} from "../types";

/** Structured backend error envelope: {"error": {"code","message","details"}} */
export class ApiError extends Error {
  code: string;
  details: unknown;
  constructor(code: string, message: string, details: unknown) {
    super(message);
    this.code = code;
    this.details = details;
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch (e) {
    throw new ApiError("NETWORK_ERROR", `Could not reach ${url}`, e);
  }
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    /* some endpoints may return empty bodies on error */
  }
  if (!res.ok) {
    const err = (body as { error?: { code: string; message: string; details: unknown } } | null)?.error;
    throw new ApiError(err?.code ?? "HTTP_ERROR", err?.message ?? res.statusText, err?.details);
  }
  return body as T;
}

/** Detection Backend (detection-backend-app/detection_backend.py) client. */
export class DetectionBackendClient {
  constructor(private baseUrl: string) {}

  health() {
    return request<{ status: string; uptime_s: number }>(`${this.baseUrl}/api/health`);
  }
  status() {
    return request<SystemStatus>(`${this.baseUrl}/api/status`);
  }
  config() {
    return request<Record<string, unknown>>(`${this.baseUrl}/api/config`);
  }
  modelMetrics() {
    return request<{ available: boolean; training_history?: unknown; note?: string }>(
      `${this.baseUrl}/api/model/metrics`
    );
  }
  latestDetection() {
    return request<DetectionResult | { detected: false; note: string }>(
      `${this.baseUrl}/api/latest_detection`
    );
  }
  tracks() {
    return request<Record<string, Track>>(`${this.baseUrl}/api/tracks`);
  }
  beamSpectrum() {
    return request<BeamSpectrumData | { angles_deg: []; power_db: []; peak_angle_deg: null }>(
      `${this.baseUrl}/api/beam_spectrum`
    );
  }
  spectrogram() {
    return request<SpectrogramData>(`${this.baseUrl}/api/spectrogram`);
  }
  private async analyzeFile(endpoint: string, file: File): Promise<DetectionResult> {
    const fd = new FormData();
    fd.append("file", file);
    return request<DetectionResult>(`${this.baseUrl}${endpoint}`, { method: "POST", body: fd });
  }
  analyzeWav(file: File) { return this.analyzeFile("/api/analyze_wav", file); }
  analyzeNpz(file: File) { return this.analyzeFile("/api/analyze_npz", file); }
  analyzeMp3(file: File) { return this.analyzeFile("/api/analyze_mp3", file); }

  modelCompare() {
    return request<ModelCompareResult>(`${this.baseUrl}/api/model/compare`);
  }
  realDatasets() {
    return request<{ datasets: RealDatasetInfo[] }>(`${this.baseUrl}/api/real_datasets`);
  }
  analyzeRealDataset(req: { dataset: string; file: string; snr_db?: number | null;
                             noise_type?: string | null; duration_s?: number }) {
    return request<DetectionResult>(`${this.baseUrl}/api/analyze_real_dataset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
  }
}

/** Signal Generator (signal-generator-app/generator_backend.py) client. */
export interface GenerateRequest {
  signal_type: string;
  frequency_hz: number;
  snr_db: number;
  noise_type: string;
  doppler: boolean;
  harmonics: boolean;
  drift_hz: number;
  bearing_deg: number;
  n_hydrophones: number;
  sampling_rate: number;
  duration_s: number;
  array_spacing_m: number;
  seed?: number;
}

export class SignalGeneratorClient {
  constructor(private baseUrl: string) {}

  generate(req: GenerateRequest) {
    return request<Record<string, unknown>>(`${this.baseUrl}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
  }
  status() {
    return request<Record<string, unknown>>(`${this.baseUrl}/api/status`);
  }
  last() {
    return request<Record<string, unknown>>(`${this.baseUrl}/api/last`);
  }
  preview(maxPoints = 2000) {
    return request<{ fs_effective_points_per_s: number; values: number[] }>(
      `${this.baseUrl}/api/preview?max_points=${maxPoints}`
    );
  }
  transmit(host: string, port: number, realtime = true) {
    return request<{ packets_sent: number; host: string; port: number }>(
      `${this.baseUrl}/api/transmit`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ host, port, realtime }) }
    );
  }
  saveWav() {
    return request<{ path: string; filename: string }>(`${this.baseUrl}/api/save_wav`, { method: "POST" });
  }
  saveArray() {
    return request<{ path: string; filename: string }>(`${this.baseUrl}/api/save_array`, { method: "POST" });
  }

  realDatasets() {
    return request<{ datasets: RealDatasetInfo[] }>(`${this.baseUrl}/api/real_datasets`);
  }
  async uploadRealDataset(file: File) {
    const fd = new FormData();
    fd.append("file", file);
    return request<{ filename: string; dataset: string; duration_s: number; sample_rate_hz: number }>(
      `${this.baseUrl}/api/real_datasets/upload`, { method: "POST", body: fd }
    );
  }
  generateWithRealBackground(req: {
    dataset: string; file?: string | null; signal_type: string; frequency_hz: number; snr_db: number;
    doppler: boolean; harmonics: boolean; drift_hz: number; bearing_deg: number; n_hydrophones: number;
    sampling_rate: number; duration_s: number; array_spacing_m: number; seed?: number;
  }) {
    return request<Record<string, unknown>>(`${this.baseUrl}/api/real/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
  }
}
