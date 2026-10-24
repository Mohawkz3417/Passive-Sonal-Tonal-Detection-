/**
 * Domain types. These mirror 18_Contracts/data_models.py field-for-field
 * (checked against the actual Python source, not guessed) so that
 * BackendDataProvider needs no invented fields, and DemoDataProvider is
 * held to the exact same shape -- see services/providers.ts.
 */

export type TrackStatus = "TENTATIVE" | "CONFIRMED" | "LOST" | "CLOSED" | "NONE";
export type ClassificationLabel = "SHIP_LIKE" | "BIOLOGICAL" | "UNKNOWN";
export type PipelineState = "READY" | "RUNNING" | "WARNING" | "ERROR" | "UNAVAILABLE";
export type DataMode = "demo" | "live";

export interface BeamSpectrumData {
  timestamp: string;
  angles_deg: number[];
  power_db: number[];
  peak_angle_deg: number | null;
}

export interface SpectrogramData {
  available: boolean;
  timestamp?: string;
  sample_rate_hz?: number;
  frequencies_hz?: number[];
  times_s?: number[];
  magnitude?: number[][];
  detection_mask?: number[][];
  seg_threshold?: number;
  note?: string;
}

/** Matches DetectionResult in data_models.py exactly. */
export interface DetectionResult {
  detected: boolean;
  timestamp: string;
  source: string;
  n_channels: number;
  bearing_available: boolean;
  estimated_bearing_deg: number | null;
  frequency_hz: number | null;
  requested_snr_db: number | null;
  measured_snr_db: number | null;
  segformer_confidence: number | null;
  classification: ClassificationLabel | null;
  classification_confidence: number | null;
  track_id: number | null;
  track_status: TrackStatus;
  bandwidth_hz: number | null;
  n_harmonics: number | null;
  array_processing_ms: number;
  model_inference_ms: number;
  classification_ms: number;
  total_ms: number;
  beam_spectrum: BeamSpectrumData | null;
  all_detections: unknown[];
  warnings: string[];
}

export interface TrackPoint {
  timestamp: string;
  bearing_deg: number | null;
  frequency_hz: number | null;
  confidence: number | null;
}

/** Matches the Track contract + /api/tracks response shape. */
export interface Track {
  track_id: number;
  status: TrackStatus;
  points: TrackPoint[];
}

/** A UI-level convenience combining a Track with its latest known
 * classification/SNR for table/card display -- not a separate backend
 * concept, just what the Contacts page needs (Part 15/53 note: this must
 * still be built FROM Track + DetectionResult, never invented separately). */
export interface Contact {
  id: string;
  track_id: number;
  status: TrackStatus;
  bearing_deg: number | null;
  frequency_hz: number | null;
  snr_db: number | null;
  classification: ClassificationLabel | null;
  confidence: number | null;
  first_detected: string;
  last_updated: string;
  bandwidth_hz: number | null;
  n_harmonics: number | null;
  track_hits: number;
  track_misses: number;
}

export interface SensorStatus {
  id: number;
  label: string;
  online: boolean;
  signal_level: number | null;
  clipping: boolean;
  warning: string | null;
}

export interface NetworkStatus {
  connected: boolean;
  protocol: string;
  host: string;
  port: number;
  packets_received: number;
  packets_dropped: number;
  packet_loss_percent: number;
  queue_overruns: number;
  sequence_number: number | null;
  sample_rate_hz: number | null;
  n_channels: number | null;
  last_packet_timestamp: string | null;
  warnings: string[];
}

export interface ModelStatus {
  architecture: string;
  checkpoint_path: string;
  loaded: boolean;
  device: string;
  training_metrics_available: boolean;
  inference_latency_ms: number | null;
}

/** One training-history row, as written by train.py to 11_Weights/metrics/*_history.json. */
export interface TrainingHistoryEpoch {
  epoch: number;
  train_loss: number;
  precision: number;
  recall: number;
  f1: number;
  iou: number;
  time_s?: number;
}

export interface ModelCompareEntry {
  architecture: string;
  available: boolean;
  checkpoint_path: string;
  checkpoint_exists: boolean;
  load_error?: string | null;
  training_history: TrainingHistoryEpoch[] | null;
  last_inference_latency_ms: number | null;
  last_detections: { frequency_hz: number; confidence: number; track_id: number; track_hits: number }[];
  last_scene_snr_db: number | null;
}

/** GET /api/model/compare -- SegFormer (primary detector) vs U-Net (baseline),
 * run on the identical waveform for a fair side-by-side (see backend docstring). */
export interface ModelCompareResult {
  segformer: ModelCompareEntry;
  unet: ModelCompareEntry;
  note: string;
}

/** One entry from GET /api/real_datasets -- an HONEST listing (0 files if not
 * configured/present, never fabricated) of ShipsEar/DeepShip/DS3500/UATD/NPOL/uploaded. */
export interface RealDatasetInfo {
  name: string;
  description: string;
  configured_path: string | null;
  folder_exists: boolean;
  file_count: number;
  files: string[];
}

export interface RawSegmentationBlob {
  frequency_hz: number;
  seg_prob: number;
  snr_db: number;
  bandwidth_hz: number;
  n_harmonics_found: number;
}

/** Additive `_raw_segmentation` field on analyze_* responses: the raw
 * SegFormer+CFAR+CCA output BEFORE tracker confirmation (see backend
 * docstring for why this matters for one-shot file analysis). */
export interface RawSegmentationSummary {
  available: boolean;
  note?: string;
  raw_blobs_found?: number;
  raw_blobs?: RawSegmentationBlob[];
  scene_snr_db?: number;
  mean_segmentation_probability?: number;
  max_segmentation_probability?: number;
}

export interface PerformanceStatus {
  array_processing_ms: number | null;
  model_inference_ms: number | null;
  classification_ms: number | null;
  total_ms: number | null;
  dropped_packets: number;
  dropped_chunks: number | null;
}

/** Matches SystemStatus (the /api/status response) exactly. */
export interface SystemStatus {
  system: { uptime_s: number; n_frames_processed: number };
  input: { mode: string; connected: boolean };
  array: { n_hydrophones: number; spacing_m: number; sound_speed_mps: number };
  network: NetworkStatus;
  sensors: SensorStatus[];
  model: ModelStatus;
  pipeline: Record<string, PipelineState>;
  performance: PerformanceStatus;
  warnings: string[];
}

export interface DetectionEvidence {
  segformer_confidence: number | null;
  frequency_hz: number | null;
  snr_db: number | null;
  bandwidth_hz: number | null;
  harmonic_count: number | null;
  bearing_deg: number | null;
  bearing_stability: number | null;
  beam_peak_strength_db: number | null;
  classification: ClassificationLabel | null;
  classification_confidence: number | null;
  track_confirmed: boolean;
}

export type EventSeverity = "info" | "warning" | "error";
export interface EventLogEntry {
  id: string;
  timestamp: string;
  severity: EventSeverity;
  message: string;
}

export const PIPELINE_STAGE_ORDER = [
  "INPUT", "ARRAY_PROCESSING", "BEAMFORMER", "DOA", "SEGFORMER",
  "POSTPROCESSING", "CLASSIFIER", "TRACKER", "ETHERNET",
] as const;

export type DemoScenarioId = "ship_contact" | "low_snr" | "biological" | "search";

export interface DemoScenarioDef {
  id: DemoScenarioId;
  label: string;
  frequency_hz: number;
  bearing_deg: number;
  snr_db: number;
  noise_type: string;
  doppler: boolean;
  harmonics: boolean;
  classification: ClassificationLabel | null;
}

export interface WsEnvelope<T = unknown> {
  type: "status" | "detection" | "track" | "beam_spectrum" | "spectrogram" | "warning" | "error";
  timestamp: string;
  data: T;
}

export type ConnectionState = "connecting" | "live" | "disconnected";
