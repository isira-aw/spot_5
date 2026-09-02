/**
 * Talking to the desk over HTTP.
 *
 * The admin token is read from the environment once. When it is absent the
 * admin surface is not rendered at all — see `hasAdminToken` — rather than
 * offering controls that would come back 401.
 */
import type {
  CycleRow,
  DeskState,
  Engine2Job,
  Engine2Models,
  EngineSignal,
  EquityPoint,
  Health,
} from "./types";

const RAW_BASE = (import.meta.env.VITE_API_BASE ?? "http://localhost:8000").replace(/\/$/, "");
const ADMIN_TOKEN = (import.meta.env.VITE_ADMIN_TOKEN ?? "").trim();

export const API_BASE = RAW_BASE;
export const hasAdminToken = ADMIN_TOKEN.length > 0;

export function websocketUrl(): string {
  const url = new URL(`${API_BASE}/ws`);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch (cause) {
    throw new ApiError(
      `Cannot reach the server at ${API_BASE}. Check that it is running.`,
      undefined,
    );
  }
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new ApiError(
      detail?.slice(0, 300) || `${response.status} ${response.statusText}`,
      response.status,
    );
  }
  return (await response.json()) as T;
}

export const api = {
  state: () => request<DeskState>("/state"),
  health: () => request<Health>("/health"),
  decisions: (limit = 20) => request<CycleRow[]>(`/decisions?limit=${limit}`),
  equity: (limit = 200) => request<EquityPoint[]>(`/equity?limit=${limit}`),
  engine2Models: () => request<Engine2Models>("/engine2/models"),
  cycle: (cycleId: string) =>
    request<{ cycle_id: string; signals: EngineSignal[] }>(
      `/cycles/${encodeURIComponent(cycleId)}`,
    ),
};

/**
 * Admin writes. Every one of these is triggered by an explicit, confirmed click
 * — never by page load, focus or reconnect.
 */
async function adminGet<T>(path: string): Promise<T> {
  if (!hasAdminToken) throw new ApiError("No admin token is configured.", 401);
  return request<T>(path, { headers: { "X-Admin-Token": ADMIN_TOKEN } });
}

async function adminPost<T>(path: string, body?: unknown): Promise<T> {
  if (!hasAdminToken) throw new ApiError("No admin token is configured.", 401);
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Admin-Token": ADMIN_TOKEN },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

export const admin = {
  killSwitch: (enabled: boolean, reason?: string) =>
    adminPost<unknown>("/admin/kill-switch", { enabled, reason }),
  runCycle: () => adminPost<unknown>("/admin/cycle/run", { autotrade: true, force: true }),
  setMode: (mode: "PAPER" | "REAL") => adminPost<unknown>("/admin/mode", { mode }),
  resetPaper: () => adminPost<unknown>("/admin/paper/reset"),
  scheduler: (action: "start" | "stop") => adminPost<unknown>(`/admin/scheduler/${action}`),

  // engine_2 is a model factory: these train and promote a model artifact. None
  // of them can place an order.
  engine2Start: (body: { job: string; walkforward?: boolean; skip_fetch?: boolean }) =>
    adminPost<Engine2Job>("/admin/engine2/job", body),
  engine2Job: () => adminGet<Engine2Job>("/admin/engine2/job"),
  engine2Rollback: (version?: string) =>
    adminPost<{ version: string; previous: string | null }>(
      `/admin/engine2/rollback${version ? `?version=${encodeURIComponent(version)}` : ""}`,
    ),
};
