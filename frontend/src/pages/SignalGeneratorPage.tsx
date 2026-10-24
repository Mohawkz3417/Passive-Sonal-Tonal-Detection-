import React, { useEffect, useMemo, useState } from "react";
import { useApp } from "../store";
import { Panel, StatusPill } from "../components/common";
import { SignalGeneratorClient, ApiError } from "../services/restClient";
import { toastBus } from "../hooks";
import type { RealDatasetInfo } from "../types";

const DEFAULTS = {
  signal_type: "Ship-like", frequency_hz: 443, snr_db: -7, noise_type: "shipping",
  doppler: true, harmonics: true, drift_hz: 0.3, bearing_deg: 37,
  n_hydrophones: 8, sampling_rate: 16000, duration_s: 10, array_spacing_m: 0.2,
};

type SourceMode = "synthetic" | "real_background";

export function SignalGeneratorPage() {
  const { settings, mode } = useApp();
  const client = useMemo(() => new SignalGeneratorClient(settings.signalGeneratorUrl), [settings.signalGeneratorUrl]);
  const [source, setSource] = useState<SourceMode>("synthetic");
  const [form, setForm] = useState(DEFAULTS);
  const [info, setInfo] = useState<Record<string, unknown> | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // -------- Real dataset / recording state --------
  const [datasets, setDatasets] = useState<RealDatasetInfo[]>([]);
  const [dsName, setDsName] = useState("");
  const [dsFile, setDsFile] = useState<string>("");         // "" = pick randomly
  const [uploadBusy, setUploadBusy] = useState(false);

  const refreshDatasets = () => client.realDatasets().then((r) => {
    setDatasets(r.datasets);
    if (!dsName) { const withFiles = r.datasets.find((d) => d.file_count > 0); if (withFiles) setDsName(withFiles.name); }
  }).catch(() => setDatasets([]));

  useEffect(() => { if (source === "real_background") refreshDatasets(); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [source]);

  const set = <K extends keyof typeof form>(key: K, value: (typeof form)[K]) => setForm((f) => ({ ...f, [key]: value }));

  const run = async (action: string, fn: () => Promise<unknown>) => {
    setBusy(action); setError(null);
    try {
      const r = await fn();
      if (action === "generate") setInfo(r as Record<string, unknown>);
      toastBus.push(`${action} succeeded`, "success");
    } catch (e) {
      const msg = e instanceof ApiError ? `${e.code}: ${e.message}` : "Could not reach Signal Generator backend";
      setError(msg);
      toastBus.push(`${action} failed`, "error");
    } finally {
      setBusy(null);
    }
  };

  const onUpload = async (f: File) => {
    setUploadBusy(true); setError(null);
    try {
      await client.uploadRealDataset(f);
      toastBus.push(`Uploaded ${f.name}`, "success");
      await refreshDatasets();
      setDsName("uploaded");
    } catch (e) {
      const msg = e instanceof ApiError ? `${e.code}: ${e.message}` : "Upload failed";
      setError(msg);
      toastBus.push("Upload failed", "error");
    } finally {
      setUploadBusy(false);
    }
  };

  const currentDatasetFiles = datasets.find((d) => d.name === dsName)?.files ?? [];

  return (
    <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", gap: 14 }}>
      <Panel title="Signal Generator Controls" actions={<StatusPill tone={mode === "live" ? "ok" : "off"}>{settings.signalGeneratorUrl}</StatusPill>}>
        <div className="flex gap-1" style={{ marginBottom: 14 }}>
          <button className="btn" style={{ background: source === "synthetic" ? "var(--accent)" : undefined, color: source === "synthetic" ? "#061014" : undefined }}
                  onClick={() => setSource("synthetic")}>Fully Synthetic</button>
          <button className="btn" style={{ background: source === "real_background" ? "var(--accent)" : undefined, color: source === "real_background" ? "#061014" : undefined }}
                  onClick={() => setSource("real_background")}>Real Recording (background)</button>
        </div>

        {source === "real_background" && (
          <div className="flex-col gap-2" style={{ marginBottom: 14, padding: 10, background: "var(--bg-inset)", borderRadius: 8 }}>
            <div className="text-tertiary" style={{ fontSize: 11.5 }}>
              Uses a REAL recording (ShipsEar/DeepShip/DS3500/UATD/NPOL — configured in{" "}
              <span className="mono">config.yaml</span> — or a file you upload here) as the array's ambient
              noise background, with a synthetic, ground-truth-known tonal (below) placed at the chosen
              bearing/frequency on top of it. Per-channel independence is approximated from one mono
              recording (see the generated result's warnings) — this is a controlled test against real
              ocean/ship noise character, not a claim of a real 8-hydrophone capture.
            </div>
            <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <Field label="Dataset">
                <select className="field w-full" value={dsName} onChange={(e) => { setDsName(e.target.value); setDsFile(""); }}>
                  <option value="">— select —</option>
                  {datasets.map((d) => (
                    <option key={d.name} value={d.name} disabled={d.file_count === 0}>
                      {d.name} ({d.file_count} file{d.file_count === 1 ? "" : "s"})
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="File (blank = random)">
                <select className="field w-full" value={dsFile} onChange={(e) => setDsFile(e.target.value)}>
                  <option value="">— random —</option>
                  {currentDatasetFiles.map((f) => <option key={f} value={f}>{f}</option>)}
                </select>
              </Field>
            </div>
            <div>
              <label className="btn" style={{ display: "inline-block", cursor: "pointer" }}>
                {uploadBusy ? "Uploading…" : "Upload a recording (.wav/.flac)"}
                <input type="file" accept=".wav,.flac" style={{ display: "none" }}
                       onChange={(e) => { const f = e.target.files?.[0]; if (f) onUpload(f); }} />
              </label>
            </div>
          </div>
        )}

        <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", gap: 10 }}>
          <Field label="Signal Type">
            <select className="field w-full" value={form.signal_type} onChange={(e) => set("signal_type", e.target.value)}>
              <option>Ship-like</option><option>Biological</option><option>Unknown</option>
            </select>
          </Field>
          {source === "synthetic" && (
            <Field label="Noise Type">
              <select className="field w-full" value={form.noise_type} onChange={(e) => set("noise_type", e.target.value)}>
                <option value="shipping">Shipping</option><option value="environmental">Environmental</option>
                <option value="biological">Biological</option><option value="background">Background</option>
              </select>
            </Field>
          )}
          <Field label="Frequency (Hz)"><input className="field w-full" type="number" value={form.frequency_hz} onChange={(e) => set("frequency_hz", Number(e.target.value))} /></Field>
          <Field label="SNR (dB)"><input className="field w-full" type="number" value={form.snr_db} onChange={(e) => set("snr_db", Number(e.target.value))} /></Field>
          <Field label="Bearing (°)"><input className="field w-full" type="number" value={form.bearing_deg} onChange={(e) => set("bearing_deg", Number(e.target.value))} /></Field>
          <Field label="Drift (Hz)"><input className="field w-full" type="number" step="0.1" value={form.drift_hz} onChange={(e) => set("drift_hz", Number(e.target.value))} /></Field>
          <Field label="Hydrophones"><input className="field w-full" type="number" value={form.n_hydrophones} onChange={(e) => set("n_hydrophones", Number(e.target.value))} /></Field>
          <Field label="Array Spacing (m)"><input className="field w-full" type="number" step="0.05" value={form.array_spacing_m} onChange={(e) => set("array_spacing_m", Number(e.target.value))} /></Field>
          <Field label="Sampling Rate (Hz)"><input className="field w-full" type="number" value={form.sampling_rate} onChange={(e) => set("sampling_rate", Number(e.target.value))} /></Field>
          <Field label="Duration (s)"><input className="field w-full" type="number" value={form.duration_s} onChange={(e) => set("duration_s", Number(e.target.value))} /></Field>
          {source === "synthetic" && <label className="flex items-center gap-1" style={{ fontSize: 12 }}><input type="checkbox" checked={form.doppler} onChange={(e) => set("doppler", e.target.checked)} /> Doppler</label>}
          <label className="flex items-center gap-1" style={{ fontSize: 12 }}><input type="checkbox" checked={form.harmonics} onChange={(e) => set("harmonics", e.target.checked)} /> Harmonics</label>
        </div>

        <div className="flex gap-1" style={{ marginTop: 16, flexWrap: "wrap" }}>
          {source === "synthetic" ? (
            <button className="btn primary" disabled={busy !== null} onClick={() => run("generate", () => client.generate(form))}>Generate</button>
          ) : (
            <button className="btn primary" disabled={busy !== null || !dsName} onClick={() => run("generate", () => client.generateWithRealBackground({
              dataset: dsName, file: dsFile || null, signal_type: form.signal_type, frequency_hz: form.frequency_hz,
              snr_db: form.snr_db, doppler: form.doppler, harmonics: form.harmonics, drift_hz: form.drift_hz,
              bearing_deg: form.bearing_deg, n_hydrophones: form.n_hydrophones, sampling_rate: form.sampling_rate,
              duration_s: form.duration_s, array_spacing_m: form.array_spacing_m,
            }))}>Generate with Real Background</button>
          )}
          <button className="btn" disabled={busy !== null} onClick={() => run("preview", () => client.preview())}>Preview</button>
          <button className="btn" disabled={busy !== null} onClick={() => run("transmit", () => client.transmit("127.0.0.1", 50001))}>Transmit</button>
          <button className="btn" disabled={busy !== null} onClick={() => run("save wav", () => client.saveWav())}>Save WAV</button>
          <button className="btn" disabled={busy !== null} onClick={() => run("save array", () => client.saveArray())}>Save Array</button>
        </div>
        {error && <div className="text-err" style={{ marginTop: 10, fontSize: 12 }}>{error}. {mode === "demo" && "Switch to LIVE mode and ensure the generator backend is running on the configured URL."}</div>}
      </Panel>

      <Panel title="Last Generation Result">
        {info ? (
          <pre className="mono" style={{ fontSize: 11, whiteSpace: "pre-wrap", background: "var(--bg-inset)", padding: 12, borderRadius: 6, maxHeight: 420, overflow: "auto" }}>
            {JSON.stringify(info, null, 2)}
          </pre>
        ) : <div className="text-tertiary" style={{ fontSize: 12.5 }}>No signal generated yet this session.</div>}
      </Panel>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label style={{ fontSize: 10.5, color: "var(--text-tertiary)", display: "block", marginBottom: 3, textTransform: "uppercase" }}>{label}</label>
      {children}
    </div>
  );
}
