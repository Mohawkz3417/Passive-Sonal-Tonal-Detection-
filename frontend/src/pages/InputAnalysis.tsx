import React, { useEffect, useMemo, useState } from "react";
import { useApp } from "../store";
import { Panel, StatusPill } from "../components/common";
import { EvidencePanel } from "../components/ContactsUI";
import type { DetectionResult, RawSegmentationSummary, RealDatasetInfo } from "../types";
import { ApiError, DetectionBackendClient } from "../services/restClient";
import { UploadCloud, FileAudio, Database } from "lucide-react";
import { toastBus } from "../hooks";

type FileKind = "wav" | "npz" | "mp3" | "real_dataset";
type AnalyzedResult = DetectionResult & {
  _raw_segmentation?: RawSegmentationSummary;
  _real_dataset_context?: { dataset: string; file: string; native_sample_rate_hz: number;
                             resampled_to_hz: number; noise_condition: string };
};

export function InputAnalysisPage() {
  const { provider, mode, settings } = useApp();
  const [kind, setKind] = useState<FileKind>("wav");
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalyzedResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  // -------- Real Dataset tab state (live-backend only -- see notes below) --------
  const detClient = useMemo(() => new DetectionBackendClient(settings.detectionBackendUrl), [settings.detectionBackendUrl]);
  const [datasets, setDatasets] = useState<RealDatasetInfo[]>([]);
  const [dsName, setDsName] = useState<string>("");
  const [dsFile, setDsFile] = useState<string>("");
  const [dsSnr, setDsSnr] = useState<string>("");
  const [dsNoise, setDsNoise] = useState("shipping");

  useEffect(() => {
    if (kind !== "real_dataset" || mode !== "live") return;
    detClient.realDatasets().then((r) => {
      setDatasets(r.datasets);
      const withFiles = r.datasets.find((d) => d.file_count > 0);
      if (withFiles) { setDsName(withFiles.name); setDsFile(withFiles.files[0]); }
    }).catch(() => setDatasets([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kind, mode, settings.detectionBackendUrl]);

  const analyzeRealDataset = async () => {
    if (!dsName || !dsFile) return;
    setBusy(true); setError(null); setResult(null);
    try {
      const r = await detClient.analyzeRealDataset({
        dataset: dsName, file: dsFile,
        snr_db: dsSnr.trim() === "" ? null : Number(dsSnr),
        noise_type: dsSnr.trim() === "" ? null : dsNoise,
      });
      setResult(r as AnalyzedResult);
      toastBus.push("Real-dataset analysis complete", "success");
    } catch (e) {
      const msg = e instanceof ApiError ? `${e.code}: ${e.message}` : e instanceof Error ? e.message : "Unknown error";
      setError(msg);
      toastBus.push("Analysis failed", "error");
    } finally {
      setBusy(false);
    }
  };

  const analyze = async (f: File) => {
    if (kind === "real_dataset") return;
    setBusy(true); setError(null); setResult(null);
    try {
      const r = await provider.analyzeFile(kind, f);
      setResult(r);
      toastBus.push("File analysis complete", "success");
    } catch (e) {
      const msg = e instanceof ApiError ? `${e.code}: ${e.message}` : e instanceof Error ? e.message : "Unknown error";
      setError(msg);
      toastBus.push("Analysis failed", "error");
    } finally {
      setBusy(false);
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) { setFile(f); analyze(f); }
  };

  return (
    <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", gap: 14 }}>
      <Panel title="Analyze Input File" actions={<StatusPill tone={mode === "demo" ? "warn" : "ok"}>{mode.toUpperCase()}</StatusPill>}>
        <div className="flex gap-1" style={{ marginBottom: 14, flexWrap: "wrap" }}>
          {(["wav", "npz", "mp3", "real_dataset"] as FileKind[]).map((k) => (
            <button key={k} className="btn" style={{ background: kind === k ? "var(--accent)" : undefined, color: kind === k ? "#061014" : undefined }}
                    onClick={() => { setKind(k); setResult(null); setError(null); }}>
              {k === "real_dataset" ? "REAL DATASET" : k.toUpperCase()}
            </button>
          ))}
        </div>

        {kind !== "real_dataset" ? (
          <>
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
              style={{
                border: `2px dashed ${dragOver ? "var(--accent)" : "var(--border)"}`, borderRadius: 10,
                padding: 36, textAlign: "center", background: dragOver ? "var(--accent-glow)" : "var(--bg-inset)",
                transition: "all 0.15s ease", cursor: "pointer",
              }}
              onClick={() => document.getElementById("file-input")?.click()}
            >
              <UploadCloud size={30} color="var(--text-tertiary)" />
              <div style={{ marginTop: 8, fontSize: 13, fontWeight: 600 }}>Drag & drop a {kind.toUpperCase()} file, or click to browse</div>
              <div className="text-tertiary" style={{ fontSize: 11, marginTop: 4 }}>
                {kind === "mp3" ? "Mono only — bearing will read N/A for MP3 input" : "Multichannel supported"}
              </div>
              <input id="file-input" type="file" style={{ display: "none" }}
                     accept={kind === "wav" ? ".wav" : kind === "npz" ? ".npz" : ".mp3"}
                     onChange={(e) => { const f = e.target.files?.[0]; if (f) { setFile(f); analyze(f); } }} />
            </div>
            {file && (
              <div className="flex items-center gap-2" style={{ marginTop: 12, fontSize: 12.5 }}>
                <FileAudio size={15} /> <span className="mono">{file.name}</span>
                {busy && <span className="text-tertiary">analyzing…</span>}
              </div>
            )}
          </>
        ) : mode !== "live" ? (
          <div className="text-tertiary" style={{ fontSize: 12.5 }}>
            Real-dataset testing needs a live Detection Backend (it reads server-side dataset
            folders). Switch to LIVE mode to use it.
          </div>
        ) : (
          <div className="flex-col gap-2">
            <div className="flex items-center gap-2" style={{ fontSize: 12 }}>
              <Database size={15} />
              <span>Datasets come from <span className="mono">config.yaml</span>'s <span className="mono">datasets:</span> section
                (ShipsEar/DeepShip/DS3500/UATD/NPOL), plus anything uploaded via the Signal Generator's
                "Real Recording" tab.</span>
            </div>
            <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <FieldSel label="Dataset" value={dsName}
                        onChange={(v) => { setDsName(v); const d = datasets.find((x) => x.name === v);
                                            setDsFile(d?.files[0] ?? ""); }}>
                {datasets.map((d) => (
                  <option key={d.name} value={d.name} disabled={d.file_count === 0}>
                    {d.name} ({d.file_count} file{d.file_count === 1 ? "" : "s"}{d.folder_exists ? "" : ", not configured"})
                  </option>
                ))}
              </FieldSel>
              <FieldSel label="File" value={dsFile} onChange={setDsFile}>
                {(datasets.find((d) => d.name === dsName)?.files ?? []).map((f) => (
                  <option key={f} value={f}>{f}</option>
                ))}
              </FieldSel>
              <FieldInput label="Inject SNR (dB, optional)" value={dsSnr} onChange={setDsSnr} placeholder="leave blank = clean" />
              <FieldSel label="Injected Noise Profile" value={dsNoise} onChange={setDsNoise}>
                <option value="shipping">Shipping</option><option value="environmental">Environmental</option>
                <option value="biological">Biological</option><option value="background">Background</option>
              </FieldSel>
            </div>
            <button className="btn primary" style={{ marginTop: 6, alignSelf: "flex-start" }}
                    disabled={busy || !dsName || !dsFile} onClick={analyzeRealDataset}>
              {busy ? "Analyzing…" : "Analyze Real Recording"}
            </button>
            {datasets.length > 0 && datasets.every((d) => d.file_count === 0) && (
              <div className="text-tertiary" style={{ fontSize: 11.5 }}>
                No real datasets are configured/present on this backend yet — this is expected until
                you point <span className="mono">config.yaml</span> at a folder (or upload a file via
                the Signal Generator), not a bug.
              </div>
            )}
          </div>
        )}
        {error && <div className="text-err" style={{ marginTop: 10, fontSize: 12 }}>{error}</div>}
        {result && (
          <div className="flex-col gap-1 mono fade-in" style={{ marginTop: 14, fontSize: 12.5 }}>
            <Row label="Detected (confirmed)" value={result.detected ? "YES" : "NO"} />
            <Row label="Bearing" value={result.bearing_available ? `${result.estimated_bearing_deg?.toFixed(1)}°` : "N/A — MONO INPUT"} />
            <Row label="Frequency" value={result.frequency_hz ? `${result.frequency_hz.toFixed(1)} Hz` : "—"} />
            <Row label="Measured SNR" value={result.measured_snr_db !== null ? `${result.measured_snr_db.toFixed(1)} dB` : "—"} />
            <Row label="Classification" value={result.classification ?? "—"} />
            {result._real_dataset_context && (
              <>
                <div style={{ height: 1, background: "var(--border-soft)", margin: "6px 0" }} />
                <Row label="Native Sample Rate" value={`${result._real_dataset_context.native_sample_rate_hz} Hz`} />
                <Row label="Resampled To" value={`${result._real_dataset_context.resampled_to_hz} Hz`} />
                <Row label="Noise Condition" value={result._real_dataset_context.noise_condition} />
              </>
            )}
            {result._raw_segmentation?.available && (
              <>
                <div style={{ height: 1, background: "var(--border-soft)", margin: "6px 0" }} />
                <div className="text-tertiary" style={{ fontSize: 10.5 }}>
                  RAW SEGFORMER OUTPUT (this call, before {result.detected ? "" : "the "}3-frame tracker confirmation)
                </div>
                <Row label="Raw Blobs Found" value={String(result._raw_segmentation.raw_blobs_found)} />
                <Row label="Max Seg. Probability" value={result._raw_segmentation.max_segmentation_probability?.toFixed(3) ?? "—"} />
                {(result._raw_segmentation.raw_blobs ?? []).map((b, i) => (
                  <Row key={i} label={`Blob ${i + 1}`} value={`${b.frequency_hz.toFixed(1)} Hz @ p=${b.seg_prob.toFixed(2)}`} />
                ))}
              </>
            )}
          </div>
        )}
      </Panel>
      <Panel title="Detection Evidence">
        <EvidencePanel detection={result} />
      </Panel>
    </div>
  );
}

function FieldSel({ label, value, onChange, children }: { label: string; value: string; onChange: (v: string) => void; children: React.ReactNode }) {
  return (
    <div>
      <label style={{ fontSize: 10.5, color: "var(--text-tertiary)", display: "block", marginBottom: 3, textTransform: "uppercase" }}>{label}</label>
      <select className="field w-full" value={value} onChange={(e) => onChange(e.target.value)}>{children}</select>
    </div>
  );
}

function FieldInput({ label, value, onChange, placeholder }: { label: string; value: string; onChange: (v: string) => void; placeholder?: string }) {
  return (
    <div>
      <label style={{ fontSize: 10.5, color: "var(--text-tertiary)", display: "block", marginBottom: 3, textTransform: "uppercase" }}>{label}</label>
      <input className="field w-full" value={value} placeholder={placeholder} onChange={(e) => onChange(e.target.value)} />
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return <div className="flex justify-between"><span className="text-secondary">{label}</span><span>{value}</span></div>;
}
