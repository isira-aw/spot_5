/**
 * The client's model of the desk.
 *
 * A snapshot replaces state wholesale; incremental frames patch it. History is
 * merged by `cycle_id`, so reapplying a snapshot after a reconnect updates rows
 * in place instead of appending duplicates.
 */
import type {
  CycleRow,
  DeskEvent,
  Decision,
  EquityPoint,
  Health,
  Portfolio,
  Quote,
  Restrictions,
  Snapshot,
  TradeFrame,
} from "../api/types";

export const MAX_HISTORY = 60;
export const MAX_EVENTS = 40;

export interface Desk {
  /** True until the first snapshot or poll lands. */
  loading: boolean;
  mode: "PAPER" | "REAL" | null;
  symbol: string;
  price: Quote | null;
  portfolio: Portfolio | null;
  decision: Decision | null;
  restrictions: Restrictions | null;
  health: Health | null;
  history: CycleRow[];
  equity: EquityPoint[];
  events: DeskEvent[];
  lastTrade: (TradeFrame & { at: string }) | null;
  cycleSeconds: number;
  /** When the server last told us anything, and when the last cycle began. */
  updatedAt: number | null;
  priceAt: number | null;
  cycleStartedAt: number | null;
  runningCycleId: string | null;
}

export const emptyDesk: Desk = {
  loading: true,
  mode: null,
  symbol: "",
  price: null,
  portfolio: null,
  decision: null,
  restrictions: null,
  health: null,
  history: [],
  equity: [],
  events: [],
  lastTrade: null,
  cycleSeconds: 900,
  updatedAt: null,
  priceAt: null,
  cycleStartedAt: null,
  runningCycleId: null,
};

export type DeskAction =
  | { kind: "snapshot"; data: Snapshot; at: number }
  | { kind: "state"; data: Snapshot | (Snapshot & { partial: true }); at: number }
  | { kind: "price"; data: Quote; at: number }
  | { kind: "portfolio"; data: Portfolio; at: number }
  | { kind: "decision"; data: Decision | null; at: number }
  | { kind: "cycle_start"; data: { cycle_id: string; started_at: string }; at: number }
  | { kind: "trade"; data: TradeFrame; at: number; ts: string }
  | { kind: "event"; data: DeskEvent; at: number }
  | { kind: "health"; data: Health; at: number }
  | { kind: "history"; data: CycleRow[]; at: number }
  | { kind: "equity"; data: EquityPoint[]; at: number };

/** Newest first, one row per cycle. */
export function mergeHistory(existing: CycleRow[], incoming: CycleRow[]): CycleRow[] {
  const byId = new Map<string, CycleRow>();
  for (const row of [...existing, ...incoming]) byId.set(row.cycle_id, row);
  return [...byId.values()]
    .sort((a, b) => Date.parse(b.started_at) - Date.parse(a.started_at))
    .slice(0, MAX_HISTORY);
}

function mergeEquity(existing: EquityPoint[], incoming: EquityPoint[]): EquityPoint[] {
  const byTs = new Map<string, EquityPoint>();
  for (const point of [...existing, ...incoming]) byTs.set(point.ts, point);
  return [...byTs.values()].sort((a, b) => Date.parse(a.ts) - Date.parse(b.ts));
}

export function reduceDesk(state: Desk, action: DeskAction): Desk {
  switch (action.kind) {
    case "snapshot":
    case "state": {
      const snap = action.data;
      return {
        ...state,
        loading: false,
        mode: snap.mode,
        symbol: snap.symbol,
        price: snap.price ?? state.price,
        portfolio: snap.portfolio ?? state.portfolio,
        decision: snap.latest_decision ?? null,
        restrictions: snap.restrictions ?? state.restrictions,
        health: snap.health ?? state.health,
        history: snap.decisions ? mergeHistory(state.history, snap.decisions) : state.history,
        equity: snap.equity ? mergeEquity(state.equity, snap.equity) : state.equity,
        cycleSeconds: snap.cycle_seconds ?? state.cycleSeconds,
        updatedAt: action.at,
        priceAt: snap.price ? action.at : state.priceAt,
        // A snapshot is authoritative: if a cycle were still running the server
        // would have said so, so stop showing one as in flight.
        runningCycleId: null,
      };
    }
    case "price":
      return { ...state, price: action.data, updatedAt: action.at, priceAt: action.at };
    case "portfolio":
      return { ...state, portfolio: action.data, updatedAt: action.at };
    case "decision": {
      if (!action.data) return { ...state, updatedAt: action.at };
      const decision = action.data;
      const row: CycleRow = {
        cycle_id: decision.cycle_id,
        started_at: decision.created_at ?? new Date(action.at).toISOString(),
        duration_ms: null,
        price: decision.entry_price,
        status: "ok",
        blocked_by: [],
        error: null,
        action: decision.action,
        confidence: decision.confidence,
        rationale: decision.rationale,
        source: decision.source ?? null,
      };
      return {
        ...state,
        decision,
        history: mergeHistory(state.history, [row]),
        runningCycleId: null,
        updatedAt: action.at,
      };
    }
    case "cycle_start":
      return {
        ...state,
        runningCycleId: action.data.cycle_id,
        cycleStartedAt: action.at,
        updatedAt: action.at,
      };
    case "trade":
      return {
        ...state,
        lastTrade: { ...action.data, at: action.ts },
        updatedAt: action.at,
      };
    case "event":
      return {
        ...state,
        events: [action.data, ...state.events].slice(0, MAX_EVENTS),
        runningCycleId:
          action.data.category === "cycle" ? null : state.runningCycleId,
        updatedAt: action.at,
      };
    case "health":
      return { ...state, health: action.data, updatedAt: action.at };
    case "history":
      return { ...state, history: mergeHistory(state.history, action.data) };
    case "equity":
      return { ...state, equity: mergeEquity(state.equity, action.data) };
    default:
      return state;
  }
}
