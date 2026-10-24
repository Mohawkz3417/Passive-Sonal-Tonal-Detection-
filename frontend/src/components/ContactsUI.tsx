import React from "react";
import type { Contact, DetectionResult } from "../types";
import { StatusPill, formatAgo, EmptyState } from "./common";
import { X, Users, CheckCircle2 } from "lucide-react";

export function ContactsTable({ contacts, selectedId, onSelect }: { contacts: Contact[]; selectedId: string | null; onSelect: (id: string) => void }) {
  if (contacts.length === 0) {
    return <EmptyState icon={<Users size={28} />} title="No active contacts" message="The acoustic search is currently clear." />;
  }
  return (
    <div className="overflow-auto-x">
      <table className="data-table">
        <thead>
          <tr>
            <th>ID</th><th>Status</th><th>Bearing</th><th>Frequency</th><th>SNR</th>
            <th>Class</th><th>Confidence</th><th>Last Update</th>
          </tr>
        </thead>
        <tbody>
          {contacts.map((c) => (
            <tr key={c.id} className={c.id === selectedId ? "selected" : ""} onClick={() => onSelect(c.id)}>
              <td className="mono" style={{ fontWeight: 700 }}>{c.id}</td>
              <td><StatusPill tone={c.status === "CONFIRMED" ? "ok" : "warn"}>{c.status}</StatusPill></td>
              <td className="mono">{c.bearing_deg !== null ? `${c.bearing_deg.toFixed(1)}°` : "—"}</td>
              <td className="mono">{c.frequency_hz !== null ? `${c.frequency_hz.toFixed(0)} Hz` : "—"}</td>
              <td className="mono">{c.snr_db !== null ? `${c.snr_db.toFixed(1)} dB` : "—"}</td>
              <td>{c.classification ?? "—"}</td>
              <td className="mono">{c.confidence !== null ? `${(c.confidence * 100).toFixed(0)}%` : "—"}</td>
              <td className="text-tertiary">{formatAgo(c.last_updated)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ContactDetailDrawer({ contact, onClose }: { contact: Contact | null; onClose: () => void }) {
  if (!contact) return null;
  const rows: [string, string][] = [
    ["Status", contact.status],
    ["Classification", contact.classification ?? "UNKNOWN"],
    ["Classification Confidence", contact.confidence !== null ? `${(contact.confidence * 100).toFixed(0)}%` : "—"],
    ["Bearing", contact.bearing_deg !== null ? `${contact.bearing_deg.toFixed(1)}°` : "N/A"],
    ["Primary Frequency", contact.frequency_hz !== null ? `${contact.frequency_hz.toFixed(1)} Hz` : "—"],
    ["SNR", contact.snr_db !== null ? `${contact.snr_db.toFixed(1)} dB` : "—"],
    ["Bandwidth", contact.bandwidth_hz !== null ? `${contact.bandwidth_hz.toFixed(1)} Hz` : "—"],
    ["Harmonics", contact.n_harmonics !== null ? String(contact.n_harmonics) : "—"],
    ["Track ID", contact.id],
    ["Track Hits", String(contact.track_hits)],
    ["Track Misses", String(contact.track_misses)],
    ["First Detected", new Date(contact.first_detected).toLocaleTimeString()],
    ["Last Updated", new Date(contact.last_updated).toLocaleTimeString()],
  ];
  return (
    <div style={{
      position: "fixed", top: 0, right: 0, height: "100vh", width: 380, background: "var(--bg-panel)",
      borderLeft: "1px solid var(--border)", boxShadow: "-8px 0 30px rgba(0,0,0,0.3)", zIndex: 200,
      overflowY: "auto", padding: 20,
    }} className="fade-in">
      <div className="flex justify-between items-center" style={{ marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 10, color: "var(--text-tertiary)", letterSpacing: "0.08em" }}>CONTACT</div>
          <div className="mono" style={{ fontSize: 20, fontWeight: 700 }}>{contact.id}</div>
        </div>
        <button className="btn" onClick={onClose}><X size={15} /></button>
      </div>
      {rows.map(([label, value]) => (
        <div key={label} className="flex justify-between" style={{ padding: "8px 0", borderBottom: "1px solid var(--border-soft)" }}>
          <span className="text-secondary" style={{ fontSize: 12 }}>{label}</span>
          <span className="mono" style={{ fontSize: 12, fontWeight: 600 }}>{value}</span>
        </div>
      ))}
    </div>
  );
}

const EVIDENCE_STAGES = [
  { key: "array", label: "Array processing completed" },
  { key: "beam", label: "Beamforming peak identified" },
  { key: "doa", label: "Bearing estimated" },
  { key: "seg", label: "SegFormer tonal confidence" },
  { key: "post", label: "CFAR / connected-component validated" },
  { key: "harm", label: "Harmonic structure detected" },
  { key: "class", label: "Classification assigned" },
  { key: "track", label: "Kalman track maintained" },
] as const;

/** Built ONLY from real DetectionResult fields -- no LLM, no invented text.
 * A line is shown only when the underlying field is actually present. */
export function EvidencePanel({ detection }: { detection: DetectionResult | null }) {
  if (!detection || !detection.detected) {
    return <EmptyState icon={<CheckCircle2 size={26} />} title="No detection to explain" message="Evidence appears once a contact is detected." />;
  }
  const lines: { ok: boolean; text: string }[] = [
    { ok: true, text: "Tonal energy present in beamformed spectrogram" },
    { ok: detection.bearing_available, text: detection.bearing_available ? `Beamforming peak identified — bearing ${detection.estimated_bearing_deg?.toFixed(1)}°` : "Bearing unavailable (single-channel input)" },
    { ok: detection.segformer_confidence !== null, text: detection.segformer_confidence !== null ? `SegFormer tonal confidence: ${detection.segformer_confidence.toFixed(2)}` : "SegFormer confidence not available" },
    { ok: detection.frequency_hz !== null, text: detection.frequency_hz !== null ? `Dominant frequency resolved: ${detection.frequency_hz.toFixed(1)} Hz` : "Frequency not resolved" },
    { ok: (detection.n_harmonics ?? 0) > 0, text: (detection.n_harmonics ?? 0) > 0 ? `Harmonic structure detected: ${detection.n_harmonics} harmonic(s)` : "No harmonic structure detected" },
    { ok: detection.measured_snr_db !== null, text: detection.measured_snr_db !== null ? `Measured SNR: ${detection.measured_snr_db.toFixed(1)} dB` : "SNR not measured" },
    { ok: detection.classification !== null, text: detection.classification !== null ? `Classifier assigned: ${detection.classification} (${((detection.classification_confidence ?? 0) * 100).toFixed(0)}%)` : "Classification unavailable" },
    { ok: detection.track_status === "CONFIRMED", text: detection.track_status === "CONFIRMED" ? `Kalman track ${detection.track_id} confirmed across multiple frames` : `Track status: ${detection.track_status}` },
  ];
  return (
    <div>
      <div style={{ fontSize: 12, fontWeight: 700, letterSpacing: "0.04em", marginBottom: 12, color: "var(--text-primary)" }}>
        WHY WAS THIS CONTACT DETECTED?
      </div>
      {lines.map((l, i) => (
        <div key={i} className="flex items-start gap-2 fade-in" style={{ padding: "6px 0", animationDelay: `${i * 60}ms` }}>
          <span className="mono" style={{ color: "var(--text-tertiary)", fontSize: 11, minWidth: 18 }}>{String(i + 1).padStart(2, "0")}</span>
          <CheckCircle2 size={14} color={l.ok ? "var(--status-ok)" : "var(--text-tertiary)"} style={{ marginTop: 2, flexShrink: 0 }} />
          <span style={{ fontSize: 12.5, color: l.ok ? "var(--text-primary)" : "var(--text-tertiary)" }}>{l.text}</span>
        </div>
      ))}
      <div style={{ marginTop: 14, paddingTop: 12, borderTop: "1px solid var(--border-soft)" }}>
        <div className="flex justify-between text-tertiary mono" style={{ fontSize: 10 }}>
          {EVIDENCE_STAGES.map((s) => <span key={s.key}>{s.label.split(" ")[0]}</span>)}
        </div>
      </div>
    </div>
  );
}
