import type { WsEnvelope, ConnectionState } from "../types";

type Listener = (msg: WsEnvelope) => void;
type StateListener = (state: ConnectionState) => void;

/**
 * Thin wrapper around the backend's /ws/live socket. Backend envelope is
 * {"type","timestamp","data"} (verified against detection_backend.py) --
 * this class passes it straight through rather than reshaping it, per
 * "do not send/assume incompatible structures."
 */
export class SonarWebSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private listeners = new Set<Listener>();
  private stateListeners = new Set<StateListener>();
  private retryDelay = 1000;
  private readonly maxRetryDelay = 15000;
  private closedByUser = false;
  private lastMessageAt: number | null = null;
  private state: ConnectionState = "disconnected";

  constructor(url: string) {
    this.url = url;
  }

  connect() {
    this.closedByUser = false;
    this.open();
  }

  private open() {
    this.setState("connecting");
    try {
      this.ws = new WebSocket(this.url);
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.ws.onopen = () => {
      this.retryDelay = 1000;
      this.setState("live");
    };
    this.ws.onmessage = (ev) => {
      this.lastMessageAt = Date.now();
      try {
        const msg = JSON.parse(ev.data) as WsEnvelope;
        this.listeners.forEach((l) => l(msg));
      } catch {
        /* ignore malformed frame */
      }
    };
    this.ws.onclose = () => {
      this.setState("disconnected");
      if (!this.closedByUser) this.scheduleReconnect();
    };
    this.ws.onerror = () => {
      this.ws?.close();
    };
  }

  private scheduleReconnect() {
    setTimeout(() => {
      if (!this.closedByUser) this.open();
    }, this.retryDelay);
    this.retryDelay = Math.min(this.retryDelay * 1.7, this.maxRetryDelay);
  }

  reconnectNow() {
    this.retryDelay = 1000;
    this.ws?.close();
    this.open();
  }

  disconnect() {
    this.closedByUser = true;
    this.ws?.close();
    this.setState("disconnected");
  }

  onMessage(listener: Listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  onStateChange(listener: StateListener) {
    this.stateListeners.add(listener);
    return () => this.stateListeners.delete(listener);
  }

  getLastMessageAt() {
    return this.lastMessageAt;
  }

  private setState(s: ConnectionState) {
    this.state = s;
    this.stateListeners.forEach((l) => l(s));
  }

  getState() {
    return this.state;
  }
}
