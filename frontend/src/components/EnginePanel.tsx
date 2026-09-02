/**
 * What the three engines said, and whether they agree.
 *
 * Disagreement is the interesting case, so it is stated in words at the top and
 * the dissenting engine carries a rule down its left edge. An engine that is
 * down is a first-class outcome here, not an empty row.
 */
import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { EngineSignal } from "../api/types";
import { duration, ratio } from "../lib/format";
import { Meter, Note, SectionHead, Skeleton } from "./primitives";

const ENGINE_ROLE: Record<string, string> = {
  engine_1: "market context",
  engine_2: "ML forecast",
  engine_3: "self-trained risk",
};

export function EnginePanel({ cycleId, loading }: { cycleId: string | null; loading: boolean }) {
  const [signals, setSignals] = useState<EngineSignal[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  useEffect(() => {
    if (!cycleId) {
      setSignals(null);
      return;
    }
    let cancelled = false;
    setPending(true);
    api
      .cycle(cycleId)
      .then((detail) => {
        if (cancelled) return;
        setSignals(detail.signals ?? []);
        setError(null);
      })
      .catch((cause: unknown) => {
        if (cancelled) return;
        setError(cause instanceof Error ? cause.message : "Could not load the engine signals.");
      })
      .finally(() => !cancelled && setPending(false));
    return () => {
      cancelled = true;
    };
  }, [cycleId]);

  const consensus = describeConsensus(signals);

  return (
    <section aria-labelledby="engines-heading">
      <SectionHead
        title="Engines"
        id="engines-heading"
        meta={cycleId ?? undefined}
        aside={
          signals ? (
            <span className="shrink-0 text-[11px] text-ink-3">
              {signals.filter((s) => s.ok).length} of {signals.length} answered
            </span>
          ) : null
        }
      />

      {loading || (pending && !signals) ? (
        <div>
          <div className="py-3">
            <Skeleton w="48%" h="12px" />
          </div>
          {[0, 1, 2].map((row) => (
            <div
              key={row}
              className="grid grid-cols-[150px_96px_1fr] gap-4 border-t border-line py-4"
            >
              <div className="space-y-1.5">
                <Skeleton w="72px" h="12px" />
                <Skeleton w="88px" h="10px" />
                <Skeleton w="46px" h="16px" />
              </div>
              <div className="space-y-1.5">
                <Skeleton w="100%" h="4px" />
                <Skeleton w="34px" h="11px" />
              </div>
              <div className="space-y-1.5">
                <Skeleton w="90%" h="11px" />
                <Skeleton w="72%" h="11px" />
              </div>
            </div>
          ))}
        </div>
      ) : error ? (
        <Note title="Engine detail unavailable." tone="error">
          {error} The decision above still stands — it was made with whatever the engines
          returned at the time.
        </Note>
      ) : !cycleId || !signals ? (
        <Note title="No engine readings yet.">
          Each cycle asks engine_1 for market context and engine_2 for a forecast, then has
          engine_3 score the setup against the desk's own trade history. Their answers appear
          here after the first cycle.
        </Note>
      ) : signals.length === 0 ? (
        <Note title="This cycle recorded no engine signals.">
          It ended before the engines were consulted — a protective exit or a failed price
          lookup does that by design.
        </Note>
      ) : (
        <>
          {consensus ? (
            <p className="flex flex-wrap items-center gap-3 py-3 text-[13px] text-ink-2">
              <span
                className={`inline-flex items-center rounded-r2 border px-1.5 py-px text-[11px] font-semibold tracking-[0.05em] ${
                  consensus.agree
                    ? "border-gain/40 bg-gain-soft text-gain"
                    : "border-warn/40 bg-warn-soft text-warn"
                }`}
              >
                {consensus.label}
              </span>
              {consensus.sentence}
            </p>
          ) : null}

          <div>
            {signals.map((signal) => (
              <EngineRow
                key={signal.engine}
                signal={signal}
                dissenting={consensus?.dissenter === signal.engine}
              />
            ))}
          </div>
        </>
      )}
    </section>
  );
}

function EngineRow({ signal, dissenting }: { signal: EngineSignal; dissenting: boolean }) {
  const direction = (signal.direction || "NEUTRAL").toUpperCase();
  const tone = !signal.ok
    ? "dead"
    : direction === "UP"
      ? "up"
      : direction === "DOWN"
        ? "down"
        : "flat";
  const chip = {
    up: "border-gain/40 bg-gain-soft text-gain",
    down: "border-loss/40 bg-loss-soft text-loss",
    flat: "border-line-strong text-ink-3",
    dead: "border-warn/40 bg-warn-soft text-warn",
  }[tone];

  return (
    <div
      className={`grid grid-cols-1 items-start gap-2 border-t border-line py-4 sm:grid-cols-[150px_96px_1fr] sm:gap-4 ${
        dissenting ? "border-l-2 border-l-warn pl-3 -ml-3" : ""
      }`}
    >
      <div className="flex flex-col gap-1">
        <b className="text-[13.5px] font-semibold">{signal.engine}</b>
        <span className="text-[11.5px] text-ink-3">{ENGINE_ROLE[signal.engine] ?? "engine"}</span>
        <span
          className={`inline-flex w-fit items-center rounded-r2 border px-1.5 py-px text-[11px] font-semibold tracking-[0.05em] ${chip}`}
        >
          {signal.ok ? titleCase(direction) : "No answer"}
        </span>
      </div>

      <div>
        <Meter
          value={signal.ok ? signal.confidence : 0}
          tone={tone === "up" ? "gain" : tone === "down" ? "loss" : "neutral"}
          label={`${signal.engine} confidence ${ratio(signal.confidence)}`}
        />
        <p className="num mt-1.5 text-[12px]">{ratio(signal.confidence)}</p>
        <p className="text-[11px] text-ink-3">
          {[
            signal.latency_ms ? duration(signal.latency_ms) : null,
            signal.source || null,
            signal.stale ? "cached" : null,
          ]
            .filter(Boolean)
            .join(" · ")}
        </p>
      </div>

      <ul className="space-y-1">
        {!signal.ok ? (
          <li className="text-[13px] text-warn">
            {signal.error ?? "no answer"} — the agent was told this engine was unavailable and
            decided without it.
          </li>
        ) : signal.reasons?.length ? (
          signal.reasons.map((reason) => (
            <li key={reason} className="relative pl-3.5 text-[13px] text-ink-2">
              <span className="absolute left-0 top-[0.62em] h-px w-1.5 bg-line-strong" />
              {reason}
            </li>
          ))
        ) : (
          <li className="text-[13px] text-ink-3">No reasons recorded for this reading.</li>
        )}
      </ul>
    </div>
  );
}

/** Do they agree? Say so in a sentence, and name the odd one out. */
function describeConsensus(signals: EngineSignal[] | null) {
  if (!signals?.length) return null;
  const live = signals.filter((s) => s.ok);
  if (live.length === 0) {
    return {
      agree: false,
      label: "All down",
      sentence: "No engine answered this cycle.",
      dissenter: null as string | null,
    };
  }
  const directions = new Map<string, string[]>();
  for (const signal of live) {
    const key = (signal.direction || "NEUTRAL").toUpperCase();
    directions.set(key, [...(directions.get(key) ?? []), signal.engine]);
  }
  const groups = [...directions.entries()].sort((a, b) => b[1].length - a[1].length);
  const majority = groups[0];
  if (!majority) return null;

  const down = signals.length - live.length;
  const downNote = down
    ? ` ${down === 1 ? "One engine" : `${down} engines`} did not answer.`
    : "";

  if (groups.length === 1) {
    return {
      agree: down === 0,
      label: down ? `Partial · ${titleCase(majority[0])}` : `Agreed · ${titleCase(majority[0])}`,
      sentence:
        live.length === 1
          ? `${live[0]!.engine} is the only engine reporting, reading ${titleCase(majority[0])}.${downNote}`
          : `All ${live.length} engines read this the same way.${downNote}`,
      dissenter: null as string | null,
    };
  }
  const minority = groups[groups.length - 1];
  const dissenter = minority && minority[1].length === 1 ? minority[1][0]! : null;
  return {
    agree: false,
    label: `Split ${majority[1].length} / ${live.length - majority[1].length}`,
    sentence:
      (dissenter
        ? `${dissenter} is the odd one out — it reads ${titleCase(minority![0])} while the rest say ${titleCase(majority[0])}.`
        : `The engines disagree: ${groups.map(([dir, names]) => `${names.join(", ")} ${titleCase(dir)}`).join("; ")}.`) + downNote,
    dissenter,
  };
}

function titleCase(word: string): string {
  return word.charAt(0) + word.slice(1).toLowerCase();
}
