/** What the money is doing. Checked, not read — so: figures, tabular, quiet. */
import type { Desk } from "../live/reducer";
import { money, percent, qty, ratio, signedMoney } from "../lib/format";
import { Eyebrow, Note, SectionHead, Skeleton, StaleTag } from "./primitives";

export function PortfolioPanel({
  desk,
  priceStale,
  error,
}: {
  desk: Desk;
  priceStale: boolean;
  error: string | null;
}) {
  const portfolio = desk.portfolio;
  const position = portfolio?.position ?? null;
  const returnPct =
    portfolio && portfolio.equity && portfolio.peak_equity
      ? ((portfolio.equity - startingEquity(portfolio.equity, portfolio.realized_pnl)) /
          Math.max(1, startingEquity(portfolio.equity, portfolio.realized_pnl))) *
        100
      : null;

  return (
    <section aria-labelledby="portfolio-heading">
      <SectionHead
        title="Portfolio"
        id="portfolio-heading"
        aside={
          <span className="shrink-0 text-[11px] text-ink-3">
            {desk.mode === "REAL" ? "live book" : "paper book"}
          </span>
        }
      />

      {error && !portfolio ? (
        <Note title="Portfolio unavailable." tone="error">
          {error} The figures are hidden rather than shown as zero — a zero here would read as
          a wiped book.
        </Note>
      ) : (
        <dl className="grid grid-cols-2 gap-x-6 gap-y-4">
          <div className="col-span-2 flex min-h-[46px] flex-col gap-0.5">
            <dt>
              <Eyebrow>Equity</Eyebrow>
            </dt>
            <dd
              className={`num text-[30px] tracking-[-0.03em] transition-colors duration-[420ms] ease-desk ${
                priceStale ? "text-ink-3" : "text-ink"
              }`}
              aria-live="polite"
              aria-atomic="true"
            >
              {portfolio ? money(portfolio.equity) : <Skeleton w="9ch" />}
            </dd>
            {/* Fixed height: the stale tag appears and disappears here and must
                not move the sections below it. */}
            <dd className="flex h-[18px] items-center gap-2 text-[12px] text-ink-3">
              {portfolio ? (
                <>
                  <span>
                    {returnPct === null ? "" : `${signedMoney(returnPct, 2)} % since inception`}
                  </span>
                  {priceStale ? <StaleTag>priced late</StaleTag> : null}
                </>
              ) : (
                <Skeleton w="14ch" />
              )}
            </dd>
          </div>

          <Figure label="Cash" value={portfolio ? money(portfolio.cash) : null} />
          <Figure
            label="Position"
            value={
              !portfolio
                ? null
                : position
                  ? `${qty(position.qty)} @ ${money(position.avg_entry_price)}`
                  : "flat"
            }
            muted={!position}
          />
          <Figure
            label="Unrealised"
            value={portfolio ? signedMoney(portfolio.unrealized_pnl) : null}
            tone={toneOf(portfolio?.unrealized_pnl)}
          />
          <Figure
            label="Realised"
            value={portfolio ? signedMoney(portfolio.realized_pnl) : null}
            tone={toneOf(portfolio?.realized_pnl)}
          />
          <Figure
            label="Max drawdown"
            value={portfolio ? percent(portfolio.max_drawdown_pct) : null}
          />
          <Figure
            label="Win rate"
            value={
              portfolio
                ? portfolio.total_trades > 0
                  ? `${percent(portfolio.win_rate * (portfolio.win_rate <= 1 ? 100 : 1), 1)}`
                  : "—"
                : null
            }
            hint={
              portfolio
                ? portfolio.total_trades > 0
                  ? `${portfolio.total_trades} trades`
                  : "no closed trades yet"
                : undefined
            }
          />
          {position ? (
            <>
              <Figure label="Stop" value={money(position.stop_price)} />
              <Figure label="Target" value={money(position.target_price)} />
              <Figure
                label="Open risk"
                value={portfolio ? percent(portfolio.open_risk_pct) : null}
              />
              <Figure label="Cycles held" value={ratio(position.bars_held, 0)} />
            </>
          ) : null}
        </dl>
      )}
    </section>
  );
}

function toneOf(value: number | undefined): "gain" | "loss" | undefined {
  if (typeof value !== "number" || !Number.isFinite(value) || value === 0) return undefined;
  return value > 0 ? "gain" : "loss";
}

function Figure({
  label,
  value,
  tone,
  hint,
  muted,
}: {
  label: string;
  value: string | null;
  tone?: "gain" | "loss";
  hint?: string;
  muted?: boolean;
}) {
  const colour = tone === "gain" ? "text-gain" : tone === "loss" ? "text-loss" : "text-ink";
  return (
    <div className="flex min-h-[64px] flex-col gap-0.5">
      <dt>
        <Eyebrow>{label}</Eyebrow>
      </dt>
      <dd
        className={`num text-[19px] tracking-[-0.02em] transition-colors duration-[420ms] ease-desk ${
          muted ? "text-ink-3" : colour
        }`}
      >
        {value === null ? <Skeleton w="7ch" /> : value}
      </dd>
      <dd className="h-[18px] text-[12px] text-ink-3">{hint ?? ""}</dd>
    </div>
  );
}

/** Equity minus what trading added is where the book started. */
function startingEquity(equity: number, realized: number): number {
  const start = equity - realized;
  return Number.isFinite(start) && start > 0 ? start : equity || 1;
}
