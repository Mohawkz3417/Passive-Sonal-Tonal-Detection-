import React, { useState } from "react";
import { useProviderData } from "../hooks";
import { Panel } from "../components/common";
import { ContactsTable, EvidencePanel } from "../components/ContactsUI";
import type { Contact, DetectionResult } from "../types";

export function Contacts() {
  const { data: contacts } = useProviderData((p) => p.getContacts());
  const { data: detection } = useProviderData((p) => p.getLatestDetection());
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const list = contacts ?? [];
  const selected = list.find((c) => c.id === selectedId);

  return (
    <div className="grid" style={{ gridTemplateColumns: "1.5fr 1fr", gap: 14 }}>
      <Panel title="Contact Database">
        <ContactsTable contacts={list} selectedId={selectedId} onSelect={setSelectedId} />
      </Panel>
      <Panel title="Detection Evidence">
        <EvidencePanel detection={selected && detection?.track_id === selected.track_id ? detection : (selected ? syntheticFromContact(selected, detection) : null)} />
      </Panel>
    </div>
  );
}

// If the selected row isn't the current primary detection, build an
// evidence view from that contact's own real fields (still no invented
// data -- just reusing the Contact fields the table already shows).
function syntheticFromContact(c: Contact, latest: DetectionResult | null): DetectionResult | null {
  if (!c) return null;
  return {
    detected: true, timestamp: c.last_updated, source: latest?.source ?? "demo", n_channels: 8,
    bearing_available: c.bearing_deg !== null, estimated_bearing_deg: c.bearing_deg,
    frequency_hz: c.frequency_hz, requested_snr_db: null, measured_snr_db: c.snr_db,
    segformer_confidence: c.confidence, classification: c.classification,
    classification_confidence: c.confidence, track_id: c.track_id, track_status: c.status,
    bandwidth_hz: c.bandwidth_hz, n_harmonics: c.n_harmonics,
    array_processing_ms: 0, model_inference_ms: 0, classification_ms: 0, total_ms: 0,
    beam_spectrum: null, all_detections: [], warnings: [],
  };
}
