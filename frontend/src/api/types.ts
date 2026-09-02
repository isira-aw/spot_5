/** The server's shapes, as verified against a live instance. */

export type Mode = "PAPER" | "REAL";
export type Action = "BUY" | "SELL" | "HOLD";

export interface Quote {
  symbol?: string;
  price: number | null;
  source: string | null;
  age_s: number | null;
  ok: boolean;
  error: string | null;
}

export interface Position {
  symbol?: string;
  qty?: number;
  avg_entry_price?: number;
  stop_price?: number | null;
  target_price?: number | null;
  bars_held?: number;
  value?: number;
  unrealized_pnl?: number;
}

export interface Portfolio {
  mode: Mode;
  cash: number;
  equity: number;
  position: Position | null;
  last_price: number | null;
  realized_pnl: number;
  unrealized_pnl: number;
  trades_today: number;
  realized_pnl_today: number;
  peak_equity: number;
  max_drawdown_pct: number;
  open_risk_pct: number;
  win_rate: number;
  total_trades: number;
}

export interface Decision {
  id?: number;
  cycle_id: string;
  mode?: Mode;
  symbol?: string;
  action: Action;
  confidence: number;
  size_pct: number;
  size_quote: number;
  entry_price: number | null;
  stop_price: number | null;
  target_price: number | null;
  time_horizon: string | null;
  rationale: string;
  change_my_mind: string[] | null;
  key_risks?: string[] | null;
  engine_agreement?: string | null;
  source?: string | null;
  degraded?: boolean;
  created_at?: string;
}

export interface Restrictions {
  max_position_pct: number;
  max_capital_at_risk_pct: number;
  max_trades_per_day: number;
  max_daily_loss_pct: number;
  max_open_positions: number;
  min_confidence: number;
  min_order_quote: number;
  allowed_actions: string[];
  kill_switch: boolean;
  allow_new_entries: boolean;
}

export interface CycleRow {
  cycle_id: string;
  started_at: string;
  duration_ms: number | null;
  price: number | null;
  status: string;
  blocked_by: string[] | null;
  error: string | null;
  action: Action | null;
  confidence: number | null;
  rationale: string | null;
  source: string | null;
}

export interface EquityPoint {
  ts: string;
  equity: number;
  cash: number;
  position_value: number;
  price: number;
  drawdown_pct: number;
}

export interface EngineSignal {
  engine: string;
  ok: boolean;
  direction: "UP" | "DOWN" | "NEUTRAL" | string;
  action_hint: string;
  confidence: number;
  horizon: string;
  latency_ms: number;
  stale: boolean;
  source: string;
  error: string | null;
  reasons: string[] | null;
  levels: Record<string, unknown> | null;
  features: Record<string, unknown> | null;
  generated_at?: string;
}

export interface Health {
  ok: boolean;
  mode: Mode;
  symbol: string;
  database: { ok: boolean; latency_ms: number; url: string; outbox: number };
  knowledge_base: {
    version: string | null;
    sections: number;
    source: string;
    reloads: number;
    last_error: string | null;
  };
  risk_model: { version: string | null; kind: string; trained_on_samples: number };
  kill_switch: boolean;
  scheduler: {
    running: boolean;
    mode: Mode;
    started_at: string | null;
    lock: { name: string; held: boolean; supported: boolean };
    tasks: { name: string; interval_s: number; last_run: string | null }[];
  };
  outbox_pending: number;
}

export interface DeskState {
  mode: Mode;
  symbol: string;
  price: Quote;
  portfolio: Portfolio;
  latest_decision: Decision | null;
  restrictions: Restrictions;
  stats: Record<string, number | string>;
}

/** The /ws opening frame: /state plus what a first paint needs. */
export interface Snapshot extends DeskState {
  health: Health;
  decisions: CycleRow[];
  equity: EquityPoint[];
  cycle_seconds: number;
}

export type MessageType =
  | "snapshot"
  | "price"
  | "cycle_start"
  | "decision"
  | "trade"
  | "portfolio"
  | "event"
  | "health"
  | "ping";

export interface Envelope<T = unknown> {
  type: MessageType;
  ts: string;
  seq: number;
  data: T;
}

export interface DeskEvent {
  ts: string;
  level: string;
  category: string;
  mode: string;
  message: string;
  payload: Record<string, unknown> | null;
}

export interface TradeFrame {
  executed?: boolean;
  reasons?: string[];
  action?: Action;
  qty?: number;
  price?: number;
  [key: string]: unknown;
}

/**
 * engine_2's model factory. Training is a background job — the UI starts one and
 * polls `Engine2Job` until it leaves the `running` state.
 */
export interface Engine2Version {
  version: string;
  current: boolean;
  created_at: number;
  meta: {
    registered_at?: string;
    git?: string;
    metrics?: Record<string, number>;
    holdout?: Record<string, number | null>;
    forecaster?: Record<string, unknown>;
  };
}

export interface Engine2Drift {
  verdict?: "healthy" | "degraded" | "warming_up";
  ok?: boolean;
  n?: number;
  needed?: number;
  dir_acc?: number;
  recent_dir_acc?: number;
  pred_std?: number;
  breaches?: number;
  reasons?: string[];
  retrain_recommended?: boolean;
  model_version?: string | null;
  error?: string;
}

export interface Engine2Job {
  job?: string;
  state?: "running" | "succeeded" | "failed" | "gated" | "interrupted";
  started_at?: string;
  finished_at?: string | null;
  step?: string;
  detail?: string;
  steps?: { step: string; detail: string; at: string }[];
  error?: string | null;
  result?: {
    ok?: boolean;
    gate?: string;
    reasons?: string[];
    elapsed_s?: number;
    promote?: { promoted?: boolean; version?: string | null; reasons?: string[] };
    bars?: number;
  } | null;
}

export interface Engine2Models {
  available: boolean;
  error?: string;
  current?: { version?: string; previous?: string | null; reason?: string; promoted_at?: string };
  history?: Engine2Version[];
  retention?: number;
  drift?: Engine2Drift;
  last_cycle?: unknown;
}
