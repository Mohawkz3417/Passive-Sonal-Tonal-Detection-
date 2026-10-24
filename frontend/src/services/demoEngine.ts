import type {
  SystemStatus, DetectionResult, BeamSpectrumData, SpectrogramData, Track, TrackPoint,
  Contact, EventLogEntry, DemoScenarioId, DemoScenarioDef,
} from "../types";

export const DEMO_SCENARIOS: Record<DemoScenarioId, DemoScenarioDef> = {
  ship_contact: {
    id: "ship_contact", label: "Ship Contact", frequency_hz: 443, bearing_deg: 37, snr_db: -7,
    noise_type: "shipping", doppler: true, harmonics: true, classification: "SHIP_LIKE",
  },
  low_snr: {
    id: "low_snr", label: "Low SNR", frequency_hz: 1174, bearing_deg: -18, snr_db: -10,
    noise_type: "shipping", doppler: true, harmonics: false, classification: "SHIP_LIKE",
  },
  biological: {
    id: "biological", label: "Biological", frequency_hz: 812, bearing_deg: 61, snr_db: -9,
    noise_type: "biological", doppler: false, harmonics: false, classification: "BIOLOGICAL",
  },
  search: {
    id: "search", label: "Search (No Contact)", frequency_hz: 0, bearing_deg: 0, snr_db: -99,
    noise_type: "environmental", doppler: false, harmonics: false, classification: null,
  },
};

type Phase = "SEARCHING" | "CANDIDATE" | "PROCESSING" | "DETECTED" | "CLASSIFIED" | "TRACKING" | "CONFIRMED";

interface InternalContact {
  trackId: number;
  phase: Phase;
  phaseTick: number;
  base: DemoScenarioDef;
  bearing: number;
  frequency: number;
  snr: number;
  confidence: number;
  bandwidth: number;
  harmonics: number;
  hits: number;
  misses: number;
  firstDetected: string;
  points: TrackPoint[];
}

const PIPELINE_KEYS = [
  "INPUT", "ARRAY_PROCESSING", "BEAMFORMER", "DOA", "SEGFORMER",
  "POSTPROCESSING", "CLASSIFIER", "TRACKER", "ETHERNET",
] as const;

const BEAM_ANGLES = Array.from({ length: 91 }, (_, i) => -90 + i * 2);
const SPEC_FREQS = Array.from({ length: 64 }, (_, i) => 100 + i * 60); // ~100-3900 Hz, 64 bins
const SPEC_MAX_COLS = 120;

function nowIso() {
  return new Date().toISOString();
}
function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}
function jitter(v: number, amount: number) {
  return v + (Math.random() * 2 - 1) * amount;
}

export class DemoEngine {
  private scenarioId: DemoScenarioId = "ship_contact";
  private running = false;
  private globalTick = 0;
  private nextTrackId = 14; // matches the spec's T-014 example
  private contacts: Map<number, InternalContact> = new Map();
  private events: EventLogEntry[] = [];
  private specCols: { magnitude: number[]; mask: number[] }[] = [];
  private beamSpectrum: BeamSpectrumData;
  private listeners = new Set<() => void>();
  private startedAt = Date.now();
  private framesProcessed = 0;

  constructor() {
    this.beamSpectrum = { timestamp: nowIso(), angles_deg: BEAM_ANGLES, power_db: BEAM_ANGLES.map(() => -30), peak_angle_deg: null };
    this.seedScenario("ship_contact");
  }

  subscribe(fn: () => void) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }
  private notify() {
    this.listeners.forEach((l) => l());
  }

  start() { this.running = true; this.logEvent("info", "Demo started"); this.notify(); }
  pause() { this.running = false; this.logEvent("info", "Demo paused"); this.notify(); }
  isRunning() { return this.running; }

  reset() {
    this.globalTick = 0;
    this.contacts.clear();
    this.events = [];
    this.specCols = [];
    this.framesProcessed = 0;
    this.startedAt = Date.now();
    this.seedScenario(this.scenarioId);
    this.logEvent("info", "Demo reset to SEARCHING");
    this.notify();
  }

  setScenario(id: DemoScenarioId) {
    this.scenarioId = id;
    this.contacts.clear();
    this.specCols = [];
    this.seedScenario(id);
    this.logEvent("info", `Scenario changed: ${DEMO_SCENARIOS[id].label}`);
    this.notify();
  }
  getScenario() { return this.scenarioId; }

  private seedScenario(id: DemoScenarioId) {
    if (id === "search") return; // no contacts
    if (id === "ship_contact") {
      // Matches the spec's example scene: a primary evolving contact plus two
      // secondary contacts at different lifecycle stages, for a populated,
      // realistic-looking Contacts table rather than a single static row.
      this.spawnContact(DEMO_SCENARIOS.ship_contact, "TRACKING");
      this.spawnContact(
        { ...DEMO_SCENARIOS.biological, frequency_hz: 1174, bearing_deg: -18, snr_db: -9.1 },
        "CONFIRMED"
      );
      this.spawnContact({ ...DEMO_SCENARIOS.search, frequency_hz: 820, bearing_deg: 61, snr_db: -11.4, classification: null }, "CANDIDATE");
    } else {
      this.spawnContact(DEMO_SCENARIOS[id], "SEARCHING");
    }
  }

  private spawnContact(base: DemoScenarioDef, startPhase: Phase) {
    const id = this.nextTrackId++;
    this.contacts.set(id, {
      trackId: id, phase: startPhase, phaseTick: 0, base,
      bearing: base.bearing_deg, frequency: base.frequency_hz, snr: base.snr_db,
      confidence: startPhase === "CONFIRMED" ? 0.88 : 0.3, bandwidth: 8 + Math.random() * 10,
      harmonics: base.harmonics ? 2 + Math.floor(Math.random() * 3) : 0,
      hits: startPhase === "CONFIRMED" ? 18 : 0, misses: 0,
      firstDetected: nowIso(), points: [],
    });
  }

  private logEvent(severity: EventLogEntry["severity"], message: string) {
    this.events.unshift({ id: `${Date.now()}-${Math.random()}`, timestamp: nowIso(), severity, message });
    if (this.events.length > 200) this.events.pop();
  }

  /** Advance the whole simulation by one tick. Call this on an interval. */
  tick() {
    if (!this.running) return;
    this.globalTick++;
    this.framesProcessed++;

    for (const ic of this.contacts.values()) this.stepContact(ic);
    this.updateBeamSpectrum();
    this.updateSpectrogram();
    this.notify();
  }

  private stepContact(ic: InternalContact) {
    ic.phaseTick++;
    const transitions: Record<Phase, { next: Phase; after: number }> = {
      SEARCHING: { next: "CANDIDATE", after: 3 },
      CANDIDATE: { next: "PROCESSING", after: 2 },
      PROCESSING: { next: "DETECTED", after: 2 },
      DETECTED: { next: "CLASSIFIED", after: 1 },
      CLASSIFIED: { next: "TRACKING", after: 1 },
      TRACKING: { next: "CONFIRMED", after: 3 },
      CONFIRMED: { next: "CONFIRMED", after: Infinity },
    };
    const t = transitions[ic.phase];
    if (ic.phaseTick >= t.after && ic.phase !== "CONFIRMED") {
      ic.phase = t.next;
      ic.phaseTick = 0;
      this.onPhaseEnter(ic);
    }

    // Realistic small evolution once past pure searching -- never a static freeze.
    if (ic.phase !== "SEARCHING") {
      ic.bearing = clamp(jitter(ic.bearing, 0.4), -90, 90);
      ic.frequency = Math.max(50, jitter(ic.frequency, ic.base.doppler ? 0.6 : 0.3));
      ic.snr = jitter(ic.snr, 0.5);
      ic.confidence = clamp(jitter(ic.confidence, 0.02), 0.4, 0.98);
      if (ic.phase === "TRACKING" || ic.phase === "CONFIRMED") {
        ic.hits += 1;
        ic.points.push({ timestamp: nowIso(), bearing_deg: ic.bearing, frequency_hz: ic.frequency, confidence: ic.confidence });
        if (ic.points.length > 200) ic.points.shift();
      }
    }
  }

  private onPhaseEnter(ic: InternalContact) {
    const id = `T-${String(ic.trackId).padStart(3, "0")}`;
    switch (ic.phase) {
      case "CANDIDATE": this.logEvent("info", `Tonal candidate — ${ic.frequency.toFixed(0)} Hz`); break;
      case "PROCESSING": this.logEvent("info", `Beamforming peak — ${ic.bearing.toFixed(0)}°`); break;
      case "DETECTED": this.logEvent("info", `SegFormer confidence — ${ic.confidence.toFixed(2)}`); break;
      case "CLASSIFIED": this.logEvent("info", `Classification — ${ic.base.classification ?? "UNKNOWN"}`); break;
      case "TRACKING": this.logEvent("info", `Track ${id} created`); break;
      case "CONFIRMED": this.logEvent("info", `Track ${id} confirmed`); break;
    }
  }

  private updateBeamSpectrum() {
    const active = [...this.contacts.values()].find((c) => c.phase !== "SEARCHING");
    const peak = active ? active.bearing : null;
    const power = BEAM_ANGLES.map((a) => {
      const base = -28 + Math.random() * 2;
      if (peak === null) return base;
      const d = a - peak;
      return base + 26 * Math.exp(-(d * d) / (2 * 6 * 6)); // Gaussian lobe, ~6deg width
    });
    this.beamSpectrum = { timestamp: nowIso(), angles_deg: BEAM_ANGLES, power_db: power, peak_angle_deg: peak };
  }

  private updateSpectrogram() {
    const active = [...this.contacts.values()].find((c) => c.phase !== "SEARCHING" && c.phase !== "CANDIDATE");
    const col = { magnitude: SPEC_FREQS.map(() => -60 + Math.random() * 8), mask: SPEC_FREQS.map(() => 0) };
    if (active) {
      const harmCount = active.harmonics;
      for (let h = 0; h <= harmCount; h++) {
        const f = active.frequency * (h + 1);
        let closest = 0, bestDist = Infinity;
        SPEC_FREQS.forEach((sf, idx) => {
          const d = Math.abs(sf - f);
          if (d < bestDist) { bestDist = d; closest = idx; }
        });
        if (bestDist < 90) {
          col.magnitude[closest] = -12 + Math.random() * 4 - h * 4;
          if (active.phase === "CONFIRMED" || active.phase === "TRACKING") col.mask[closest] = 1;
        }
      }
    }
    this.specCols.push(col);
    if (this.specCols.length > SPEC_MAX_COLS) this.specCols.shift();
  }

  // --------------------------------------------------------------- snapshot getters (read-only)
  getContacts(): Contact[] {
    return [...this.contacts.values()]
      .filter((c) => c.phase !== "SEARCHING" && c.phase !== "CANDIDATE")
      .map((c) => ({
        id: `T-${String(c.trackId).padStart(3, "0")}`,
        track_id: c.trackId,
        status: c.phase === "CONFIRMED" ? "CONFIRMED" : c.phase === "TRACKING" ? "TENTATIVE" : "TENTATIVE",
        bearing_deg: c.bearing, frequency_hz: c.frequency, snr_db: c.snr,
        classification: c.base.classification, confidence: c.confidence,
        first_detected: c.firstDetected, last_updated: nowIso(),
        bandwidth_hz: c.bandwidth, n_harmonics: c.harmonics,
        track_hits: c.hits, track_misses: c.misses,
      }));
  }

  getTracks(): Record<string, Track> {
    const out: Record<string, Track> = {};
    for (const c of this.contacts.values()) {
      if (c.points.length === 0) continue;
      out[String(c.trackId)] = {
        track_id: c.trackId,
        status: c.phase === "CONFIRMED" ? "CONFIRMED" : "TENTATIVE",
        points: c.points,
      };
    }
    return out;
  }

  getPrimaryDetection(): DetectionResult {
    const primary = [...this.contacts.values()].sort((a, b) => b.confidence - a.confidence)[0];
    const detected = !!primary && primary.phase !== "SEARCHING" && primary.phase !== "CANDIDATE";
    return {
      detected, timestamp: nowIso(), source: "demo", n_channels: 8,
      bearing_available: true, estimated_bearing_deg: detected ? primary!.bearing : null,
      frequency_hz: detected ? primary!.frequency : null,
      requested_snr_db: detected ? primary!.base.snr_db : null,
      measured_snr_db: detected ? primary!.snr : null,
      segformer_confidence: detected ? primary!.confidence : null,
      classification: detected ? primary!.base.classification : null,
      classification_confidence: detected ? primary!.confidence : null,
      track_id: detected ? primary!.trackId : null,
      track_status: detected ? (primary!.phase === "CONFIRMED" ? "CONFIRMED" : "TENTATIVE") : "NONE",
      bandwidth_hz: detected ? primary!.bandwidth : null,
      n_harmonics: detected ? primary!.harmonics : null,
      array_processing_ms: 12 + Math.random() * 4, model_inference_ms: 40 + Math.random() * 15,
      classification_ms: 1 + Math.random(), total_ms: 55 + Math.random() * 15,
      beam_spectrum: this.beamSpectrum, all_detections: [], warnings: [],
    };
  }

  getBeamSpectrum(): BeamSpectrumData { return this.beamSpectrum; }

  getSpectrogram(): SpectrogramData {
    if (this.specCols.length === 0) return { available: false, note: "no data yet" };
    return {
      available: true, timestamp: nowIso(), sample_rate_hz: 16000,
      frequencies_hz: SPEC_FREQS, times_s: this.specCols.map((_, i) => i * 0.5),
      magnitude: SPEC_FREQS.map((_, fi) => this.specCols.map((c) => c.magnitude[fi])),
      detection_mask: SPEC_FREQS.map((_, fi) => this.specCols.map((c) => c.mask[fi])),
      seg_threshold: 0.5,
    };
  }

  getEvents(): EventLogEntry[] { return this.events; }

  getSystemStatus(): SystemStatus {
    const activeAny = [...this.contacts.values()].some((c) => c.phase !== "SEARCHING");
    const pipeline: Record<string, "READY" | "RUNNING"> = {};
    PIPELINE_KEYS.forEach((k) => { pipeline[k] = this.running && activeAny ? "RUNNING" : "READY"; });
    return {
      system: { uptime_s: (Date.now() - this.startedAt) / 1000, n_frames_processed: this.framesProcessed },
      input: { mode: "demo", connected: this.running },
      array: { n_hydrophones: 8, spacing_m: 0.2, sound_speed_mps: 1500 },
      network: {
        connected: this.running, protocol: "udp (simulated)", host: "demo", port: 50001,
        packets_received: this.framesProcessed * 32, packets_dropped: 0, packet_loss_percent: 0,
        queue_overruns: 0, sequence_number: this.framesProcessed, sample_rate_hz: 16000, n_channels: 8,
        last_packet_timestamp: nowIso(), warnings: [],
      },
      sensors: Array.from({ length: 8 }, (_, i) => ({
        id: i + 1, label: `H${i + 1}`, online: this.running, signal_level: this.running ? 0.4 + Math.random() * 0.3 : null,
        clipping: false, warning: null,
      })),
      model: {
        architecture: "segformer_b0 (demo)", checkpoint_path: "demo-simulated", loaded: true,
        device: "cpu", training_metrics_available: false, inference_latency_ms: 45,
      },
      pipeline,
      performance: {
        array_processing_ms: 12, model_inference_ms: 45, classification_ms: 1.2, total_ms: 58,
        dropped_packets: 0, dropped_chunks: 0,
      },
      warnings: [],
    };
  }
}
