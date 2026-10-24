# SONAR COMMAND — Frontend

Professional operator interface for the NPOL Intelligent Passive Sonar / Tonal
Detection system. Sits on top of the existing, unmodified backend
(`detection-backend-app`, `signal-generator-app`) — this frontend contains no
signal-processing logic of its own.

## Stack

React 18 + TypeScript + Vite. No CSS framework (hand-written CSS variables
for theming, kept in `src/styles/global.css`). Charts: Recharts. Icons:
lucide-react. Routing: react-router-dom.

## Running

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
npm run build      # type-checks (tsc -b) then produces dist/
npm run lint
npm run preview    # serve the production build locally
```

## Login

Demo authentication only (Part 48/9 of the design brief) — **not** production
security, and structured so a real auth service can replace it later
(`src/store.tsx`'s `login()`).

```
Operator ID: operator
Password:    sonar123
```

## Demo Mode vs Live Mode

Toggle in the header (DEMO / LIVE), or in Settings. This is the core
architectural feature of this frontend:

- **Demo Mode** (default): `DemoDataProvider` runs a deterministic state
  machine (`src/services/demoEngine.ts`) entirely client-side — no backend
  required. It produces the exact same TypeScript types
  (`DetectionResult`, `Contact`, `Track`, `SystemStatus`, ...) that a real
  backend response would, so every page is written against one shape only.
- **Live Mode**: `BackendDataProvider` calls the real REST API and
  `/ws/live` WebSocket. Configure the backend URLs in Settings (defaults:
  `http://127.0.0.1:8000` detection backend, `http://127.0.0.1:8001` signal
  generator, `ws://127.0.0.1:8000/ws/live`).

No page imports either provider directly — they all go through the
`SonarDataProvider` interface (`src/services/providers.ts`). Switching modes
never requires touching a page component.

## Running against the real backend

```bash
# terminal 1
cd ../detection-backend-app && uvicorn detection_backend:app --port 8000 --host 127.0.0.1
# terminal 2
cd ../signal-generator-app && uvicorn generator_backend:app --port 8001 --host 127.0.0.1
# terminal 3
cd frontend && npm run dev
```
Then switch the header toggle to LIVE. CORS is already configured on the
Python side for `http://localhost:5173` (`config.yaml: api.cors_origins`).

## Project structure

```
src/
  types.ts              # domain types, mirror 18_Contracts/data_models.py exactly
  config.ts              # the ONLY place backend URLs/defaults live
  store.tsx               # auth, theme, mode, provider lifecycle (React context)
  hooks.ts                 # useProviderData (polling+subscribe), toasts, clock
  services/
    restClient.ts          # typed fetch wrappers, both backend apps
    wsClient.ts              # /ws/live client with reconnect + backoff
    demoEngine.ts             # demo state machine + scenario definitions
    providers.ts               # SonarDataProvider, DemoDataProvider, BackendDataProvider
  components/               # reusable UI: layout, PPI display, charts, tables, panels
  pages/                      # one file per route
```

## Demo scenarios

`ship_contact` (default review scenario — matches the brief's example scene:
one primary SHIP_LIKE contact plus two secondary contacts at different
lifecycle stages), `low_snr`, `biological`, `search` (no active contact).
Switch via the Demo Control drawer on the Dashboard.

## What "Live Mode" does and doesn't have yet

- The backend has no dedicated `/api/contacts` or event-log endpoint.
  `BackendDataProvider.getContacts()` is a documented adapter built from
  `/api/tracks` + `/api/latest_detection` (see the comment in
  `providers.ts`) — not an invented backend concept. `getEvents()` returns
  an empty array in Live Mode rather than fabricating log lines.
- Every other page (Dashboard, Live Sonar, Contacts, Tracking, Spectrogram,
  Beamforming, Array Status, Model Status, System Health, Input Analysis)
  is wired to real, existing endpoints.

## Troubleshooting

- **"BACKEND NOT CONNECTED" in System Health while in Live Mode**: the
  detection backend isn't reachable at the configured URL, or hasn't
  finished loading the model yet (cold start can take several seconds —
  torch import + checkpoint load). Check Settings for the URL, or switch
  back to Demo Mode.
- **CORS errors in the browser console**: add your dev origin to
  `config.yaml: api.cors_origins` on the backend and restart it.
- **WebSocket shows DISCONNECTED**: the client auto-reconnects with
  exponential backoff; check the backend is actually running and that
  `websocketUrl` in Settings matches its host/port.
