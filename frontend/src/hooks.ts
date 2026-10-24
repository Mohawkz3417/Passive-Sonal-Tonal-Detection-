import { useCallback, useEffect, useRef, useState } from "react";
import { useApp } from "./store";
import type { SonarDataProvider } from "./services/providers";

/**
 * Subscribes to the active provider's change notifications AND polls on an
 * interval (covers both demo's tick-driven updates and a live backend that
 * hasn't pushed a fresh WebSocket message yet). One hook, works identically
 * regardless of which provider is active -- pages never know the difference.
 */
export function useProviderData<T>(
  fetcher: (p: SonarDataProvider) => Promise<T>,
  intervalMs = 1500
): { data: T | null; loading: boolean; error: string | null } {
  const { provider } = useApp();
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const load = useCallback(async () => {
    try {
      const result = await fetcherRef.current(provider);
      setData(result);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [provider]);

  useEffect(() => {
    setLoading(true);
    load();
    const unsub = provider.subscribe(load);
    const id = window.setInterval(load, intervalMs);
    return () => { unsub(); window.clearInterval(id); };
  }, [provider, load, intervalMs]);

  return { data, loading, error };
}

// --------------------------------------------------------------------- toasts
export interface Toast { id: string; message: string; tone: "info" | "success" | "warning" | "error"; }
type ToastListener = (toasts: Toast[]) => void;

class ToastBus {
  private toasts: Toast[] = [];
  private listeners = new Set<ToastListener>();
  push(message: string, tone: Toast["tone"] = "info") {
    const t: Toast = { id: `${Date.now()}-${Math.random()}`, message, tone };
    this.toasts = [...this.toasts, t];
    this.emit();
    setTimeout(() => this.dismiss(t.id), 4000);
  }
  dismiss(id: string) {
    this.toasts = this.toasts.filter((t) => t.id !== id);
    this.emit();
  }
  subscribe(fn: ToastListener) { this.listeners.add(fn); fn(this.toasts); return () => this.listeners.delete(fn); }
  private emit() { this.listeners.forEach((l) => l(this.toasts)); }
}
export const toastBus = new ToastBus();

export function useToasts() {
  const [toasts, setToasts] = useState<Toast[]>([]);
  useEffect(() => {
    const unsub = toastBus.subscribe(setToasts);
    return () => { unsub(); };
  }, []);
  return { toasts, dismiss: (id: string) => toastBus.dismiss(id) };
}

export function useClock() {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(id);
  }, []);
  return now;
}
