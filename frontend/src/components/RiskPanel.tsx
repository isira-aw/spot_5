/**
 * The caps the desk trades inside, and — when something was stopped — why.
 * "Blocked" is never left as a status word; it is written out as a sentence.
 */
import type { Desk } from "../live/reducer";
import { clockTime, money, percent, ratio } from "../lib/format";
import { Note, Pill, SectionHead, Skeleton } from "./primitives";

export function RiskPanel({ desk }: { desk: Desk }) {
  const rules = desk.restrictions;
  const portfolio = desk.portfolio;
  const blocked = latestBlock(desk);

  const caps = rules
    ? [
        {
          label: "Max position",
          used: positionPct(desk),
          limit: rules.max_position_pct,
          render: (used: number | null) =>
            `${used === null ? "—" : ratio(used, 1)} / ${ratio(rules.max_position_pct, 0)} %`,
        },
        {
          label: "Capital at risk",
          used: portfolio?.open_risk_pct ?? null,
          limit: rules.max_capital_at_risk_pct,
          render: (used: number | null) =>
            `${used === null ? "—" : ratio(used, 2)} / ${ratio(rules.max_capital_at_risk_pct, 2)} %`,
        },
        {
          label: "Trades today",
          used: portfolio?.trades_today ?? null,
          limit: rules.max_trades_per_day,
          render: (used: number | null) =>
            `${used === null ? "—" : ratio(used, 0)} / ${ratio(rules.max_trades_per_day, 0)}`,
        },
        {
          label: "Daily loss",
          used: dailyLossPct(desk),
          limit: rules.max_daily_loss_pct,
          render: (used: number | null) =>
            `${used === null ? "—" : ratio(used, 2)} / ${ratio(rules.max_daily_loss_pct, 2)} %`,
        },
        {
          label: "Min confidence",
          used: desk.decision?.confidence ?? null,
          limit: rules.min_confidence,
          render: (used: number | null) =>
            `${used === null ? "—" : ratio(used)} needed ${ratio(rules.min_confidence)}`,
        },
      ]
    : [];

  return (
    <section aria-labelledby="risk-heading">
      <SectionHead
        title="Risk & restrictions"
        id="risk-heading"
        aside={
          rules ? (
            <Pill tone={rules.allow_new_entries && !rules.kill_switch ? "quiet" : "armed"}>
              {rules.kill_switch
                ? "Entries blocked"
                : rules.allow_new_entries
                  ? "Entries allowed"
                  : "Entries paused"}
            </Pill>
          ) : null
        }
      />

      {!rules ? (
        desk.loading ? (
          <div className="space-y-3">
            {[0, 1, 2, 3].map((row) => (
              <div key={row} className="space-y-1.5">
                <Skeleton w="60%" h="12px" />
                <Skeleton w="100%" h="3px" />
              </div>
            ))}
          </div>
        ) : (
          <Note title="Restrictions unavailable." tone="error">
            The desk could not tell us which caps are active. Assume the last known set still
            applies and check the server.
          </Note>
        )
      ) : (
        <ul>
          {caps.map((cap) => {
            const fraction =
              cap.used === null || !cap.limit ? 0 : Math.min(1, cap.used / cap.limit);
            return (
              <li
                key={cap.label}
                className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-0.5 border-t border-line py-3 text-[13px]"
              >
                <span className="text-ink-2">{cap.label}</span>
                <span className="num">{cap.render(cap.used)}</span>
                <span className="col-span-2 h-[3px] overflow-hidden rounded-full bg-sunken">
                  <i
                    className={`block h-full transition-[width] duration-[420ms] ease-desk ${
                      fraction > 0.85 ? "bg-warn" : "bg-accent"
                    }`}
                    style={{ width: `${fraction * 100}%` }}
                  />
                </span>
              </li>
            );
          })}
        </ul>
      )}

      {rules?.kill_switch ? (
        <Blocked title="Kill switch armed — no new entries.">
          Existing positions are still managed and protective exits still fire. Nothing new
          opens until it is disarmed.
        </Blocked>
      ) : blocked ? (
        <Blocked title={`Blocked at ${blocked.time}.`}>
          {blocked.reasons.join("; ")}.{" "}
          {blocked.action ? `The agent wanted to ${blocked.action.toLowerCase()}` : "The trade"}{" "}
          {blocked.price ? `at ${money(blocked.price)} ` : ""}but the restriction above stopped
          it. The decision is still recorded.
        </Blocked>
      ) : rules ? (
        <p className="mt-4 max-w-[42ch] text-[13px] text-ink-3">
          Nothing is being blocked. An order also needs at least{" "}
          <span className="num">{money(rules.min_order_quote)}</span> in size and one of{" "}
          {rules.allowed_actions.join(", ") || "no"} actions to reach the broker.
        </p>
      ) : null}
    </section>
  );
}

function Blocked({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mt-4 rounded-r2 border-l-2 border-warn bg-warn-soft px-4 py-3 text-[13px] text-ink">
      <b className="font-semibold text-warn">{title}</b> {children}
    </div>
  );
}

function latestBlock(desk: Desk) {
  const row = desk.history.find((entry) => (entry.blocked_by?.length ?? 0) > 0);
  if (!row) return null;
  return {
    time: clockTime(row.started_at),
    reasons: row.blocked_by ?? [],
    action: row.action,
    price: row.price,
  };
}

function positionPct(desk: Desk): number | null {
  const portfolio = desk.portfolio;
  if (!portfolio || !portfolio.equity) return null;
  const value = portfolio.position?.value;
  if (typeof value !== "number" || !Number.isFinite(value)) return portfolio.position ? null : 0;
  return (value / portfolio.equity) * 100;
}

function dailyLossPct(desk: Desk): number | null {
  const portfolio = desk.portfolio;
  if (!portfolio || !portfolio.equity) return null;
  const today = portfolio.realized_pnl_today;
  if (typeof today !== "number" || !Number.isFinite(today)) return null;
  return today < 0 ? (Math.abs(today) / portfolio.equity) * 100 : 0;
}

export { percent };
