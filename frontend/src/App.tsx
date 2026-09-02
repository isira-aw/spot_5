/**
 * The desk.
 *
 * Hierarchy, in the order the eye should find things: is it alive (the status
 * bar), what did the agent decide and why (the note), what is my money doing
 * (the right-hand column), everything else.
 */
import { useCallback, useMemo, useState } from "react";

import { admin, hasAdminToken } from "./api/client";
import { ConfirmDialog, type ConfirmRequest } from "./components/Confirm";
import { Controls } from "./components/Controls";
import { DecisionPanel } from "./components/DecisionPanel";
import { EnginePanel } from "./components/EnginePanel";
import { EquityCurve } from "./components/EquityCurve";
import { History } from "./components/History";
import { PortfolioPanel } from "./components/Portfolio";
import { RiskPanel } from "./components/RiskPanel";
import { StatusBar } from "./components/StatusBar";
import { ThemeToggle } from "./components/ThemeToggle";
import { Button, Note } from "./components/primitives";
import { useLiveDesk } from "./live/useLiveDesk";
import { useElapsed, useNow } from "./lib/useNow";

/** A price is stale once it is older than twice the interval we expect it at. */
const PRICE_INTERVAL_S = 5;
const POLL_INTERVAL_S = 10;

export function App() {
  const { desk, connection, error, retryInSeconds, refresh } = useLiveDesk();
  const now = useNow();
  const [confirmRequest, setConfirmRequest] = useState<ConfirmRequest | null>(null);
  // Skeletons are for a server that is actually slow, not for one 300 ms away:
  // for the first half-second the page holds still rather than painting a
  // placeholder it is about to replace.
  const settled = useElapsed(500);
  const holding = desk.loading && !settled;

  const expectedPriceInterval = connection === "live" ? PRICE_INTERVAL_S : POLL_INTERVAL_S;

  // Age of the quote itself: what the server said, plus how long we have held it.
  const priceAgeSeconds = useMemo(() => {
    if (!desk.price || desk.priceAt === null) return null;
    const held = (now - desk.priceAt) / 1000;
    return (desk.price.age_s ?? 0) + held;
  }, [desk.price, desk.priceAt, now]);

  const priceStale =
    priceAgeSeconds !== null && priceAgeSeconds > expectedPriceInterval * 2;

  // The decision goes stale when a whole extra cycle has come and gone.
  const lastCycleAt = desk.history[0]?.started_at ?? desk.decision?.created_at ?? null;
  const decisionAgeSeconds = lastCycleAt ? (now - Date.parse(lastCycleAt)) / 1000 : null;
  const decisionStale =
    decisionAgeSeconds !== null && decisionAgeSeconds > desk.cycleSeconds * 2;
  const decisionStaleReason =
    decisionAgeSeconds === null
      ? null
      : `${Math.round(decisionAgeSeconds / 60)} min old`;

  const nextCycleSeconds = useMemo(() => {
    if (!lastCycleAt || !desk.cycleSeconds) return null;
    const due = Date.parse(lastCycleAt) + desk.cycleSeconds * 1000;
    const remaining = (due - now) / 1000;
    return remaining > 0 ? remaining : 0;
  }, [lastCycleAt, desk.cycleSeconds, now]);

  const closeConfirm = useCallback(() => setConfirmRequest(null), []);
  const showControls = hasAdminToken;

  const unreachable = connection === "offline" && desk.loading;

  return (
    <div className="min-h-screen">
      <a
        href="#desk-main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:rounded-r2 focus:border focus:border-line-strong focus:bg-surface focus:px-3 focus:py-2 focus:text-[13px]"
      >
        Skip to the agent's decision
      </a>

      <StatusBar
        desk={desk}
        connection={connection}
        retryInSeconds={retryInSeconds}
        nextCycleSeconds={nextCycleSeconds}
        priceAgeSeconds={priceAgeSeconds}
        priceStale={priceStale}
        themeControl={<ThemeToggle />}
      />

      <main id="desk-main" className="mx-auto max-w-[1320px] px-4 pb-20 pt-8 sm:px-6">
        <h1 className="sr-only">
          spot_5 — autonomous spot trading desk{desk.symbol ? `, ${desk.symbol}` : ""}
          {desk.mode ? `, ${desk.mode} mode` : ""}
        </h1>
      {/* One polite live region for the whole page: this updates constantly. */}
      <p className="sr-only" aria-live="polite" aria-atomic="true">
        {connection === "live"
          ? `Connected. Latest stance ${desk.decision?.action ?? "none yet"}.`
          : connection === "reconnecting"
            ? "The connection dropped. Reconnecting."
            : "Offline. Showing the last known state and polling for updates."}
      </p>

        {holding ? (
          <div className="min-h-[60vh]" aria-busy="true" aria-label="Loading the desk" />
        ) : unreachable ? (
          <div className="rounded-r3 border border-line bg-surface p-6 shadow-lift">
            <Note title="Cannot reach the desk." tone="error">
              {error ?? "The server is not answering."} Nothing is being traded from this
              screen, and no data has been received yet. The dashboard keeps retrying on its
              own — the next attempt is in{" "}
              <span className="num">{retryInSeconds ?? 0}</span> s.
            </Note>
            <div className="mt-4">
              <Button onClick={refresh}>Retry now</Button>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-[minmax(0,1.55fr)_minmax(0,1fr)] lg:gap-12">
            <div className="flex flex-col gap-10 lg:gap-12">
              <DecisionPanel
                desk={desk}
                connection={connection}
                stale={decisionStale}
                staleReason={decisionStaleReason}
                onRunCycle={
                  showControls
                    ? () =>
                        setConfirmRequest({
                          title: "Run the first cycle now?",
                          confirmLabel: "Run a cycle",
                          tone: "primary",
                          body: (
                            <>
                              The desk will consult the engines and the agent immediately and
                              will place an order if the agent decides to, in{" "}
                              <b>{desk.mode ?? "the current"}</b> mode.
                            </>
                          ),
                          run: async () => {
                            await admin.runCycle();
                            refresh();
                          },
                        })
                    : undefined
                }
              />

              <EnginePanel
                cycleId={desk.decision?.cycle_id ?? desk.history[0]?.cycle_id ?? null}
                loading={desk.loading}
              />

              <History rows={desk.history} loading={desk.loading} cycleSeconds={desk.cycleSeconds} />
            </div>

            <div className="flex flex-col gap-10 lg:gap-12">
              <PortfolioPanel desk={desk} priceStale={priceStale} error={error} />
              <EquityCurve
                points={desk.equity}
                loading={desk.loading}
                cycleSeconds={desk.cycleSeconds}
              />
              <RiskPanel desk={desk} />
              {showControls ? (
                <Controls desk={desk} onConfirm={setConfirmRequest} onDone={refresh} />
              ) : null}
            </div>
          </div>
        )}
      </main>

      <ConfirmDialog request={confirmRequest} onClose={closeConfirm} />
    </div>
  );
}
