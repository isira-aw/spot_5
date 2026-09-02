/**
 * The hero: what the agent decided and, at length, why.
 *
 * The rationale is the product, so it is set as prose in a reading serif rather
 * than squeezed into a field. "What would change my mind" is a numbered list
 * because those conditions really are ranked triggers.
 */
import type { Connection } from "../live/useLiveDesk";
import type { Desk } from "../live/reducer";
import { clockTime, duration, money, ratio } from "../lib/format";
import { Button, Eyebrow, Note, Pill, SectionHead, Skeleton, StaleTag } from "./primitives";

const ACTION_TONE: Record<string, string> = {
  BUY: "text-gain",
  SELL: "text-loss",
  HOLD: "text-ink",
};

export function DecisionPanel({
  desk,
  connection,
  stale,
  staleReason,
  onRunCycle,
}: {
  desk: Desk;
  connection: Connection;
  stale: boolean;
  staleReason: string | null;
  onRunCycle?: () => void;
}) {
  const decision = desk.decision;
  const gate = desk.restrictions?.min_confidence ?? null;
  const nextCycleWords = desk.cycleSeconds
    ? `every ${Math.round(desk.cycleSeconds / 60)} minutes`
    : "on its usual schedule";

  return (
    <section aria-labelledby="decision-heading">
      <SectionHead
        title="Agent decision"
        id="decision-heading"
        meta={decision?.cycle_id}
        aside={
          decision?.created_at ? (
            <span className="shrink-0 text-[11px] text-ink-3">
              {clockTime(decision.created_at)}
            </span>
          ) : null
        }
      />

      <article className="overflow-hidden rounded-r3 border border-line bg-surface shadow-lift">
        {desk.loading ? (
          <Loading />
        ) : !decision ? (
          <div className="p-6">
            <Note title="No decision yet.">
              The desk runs a cycle {nextCycleWords} and the agent's note appears here the
              moment one finishes — the action, the confidence and the reasoning in full.
              Nothing is wrong; there is simply no history yet.
            </Note>
            {onRunCycle ? (
              <div className="mt-4">
                <Button onClick={onRunCycle}>Run a cycle now</Button>
              </div>
            ) : null}
          </div>
        ) : (
          <>
            <div className="flex flex-wrap items-start gap-6 px-6 pt-6 pb-4">
              <div className="min-w-[180px] flex-1">
                <Eyebrow>Stance</Eyebrow>
                <div
                  className={`mt-1.5 font-prose text-[52px] leading-[0.95] font-medium tracking-[-0.02em] transition-colors duration-[420ms] ease-desk ${
                    stale ? "text-ink-3" : (ACTION_TONE[decision.action] ?? "text-ink")
                  }`}
                >
                  {titleCase(decision.action)}
                </div>
                {stale ? (
                  <div className="mt-2 flex items-center gap-2">
                    <StaleTag>{staleReason ?? "stale"}</StaleTag>
                    {connection !== "live" ? (
                      <Pill tone="quiet">not updating</Pill>
                    ) : null}
                  </div>
                ) : null}
              </div>

              <div className="flex min-w-[200px] flex-col gap-1.5">
                <div className="flex justify-between text-[12px] text-ink-3">
                  <span>Confidence</span>
                  <span className="num text-ink">{ratio(decision.confidence)}</span>
                </div>
                <ConfidenceSegments value={decision.confidence} gate={gate} />
                {gate !== null ? (
                  <p className="text-[11.5px] text-ink-3">
                    Entry gate is <span className="num">{ratio(gate)}</span> —{" "}
                    {decision.confidence >= gate ? "this clears it." : "this is short of it."}
                  </p>
                ) : null}
              </div>

              <dl className="flex flex-wrap gap-6">
                <Figure label="Size" value={`${ratio(decision.size_pct, 1)} %`} />
                <Figure label="Horizon" value={decision.time_horizon || "N/A"} />
              </dl>
            </div>

            <div className="px-6 pb-6">
              <div className="max-w-[66ch] font-prose text-[17.5px] leading-[1.62] text-ink">
                {splitParagraphs(decision.rationale).map((paragraph, index) => (
                  <p key={index} className={index === 0 ? "" : "mt-[0.7em]"}>
                    {paragraph}
                  </p>
                ))}
              </div>

              {decision.change_my_mind?.length ? (
                <div className="mt-6 border-t border-line pt-4">
                  <Eyebrow>What would change my mind</Eyebrow>
                  <ol className="mt-2">
                    {decision.change_my_mind.map((item, index) => (
                      <li
                        key={item}
                        className="grid grid-cols-[16px_1fr] gap-2 border-line py-1.5 text-[13.5px] text-ink-2 [&+&]:border-t [&+&]:border-dotted"
                      >
                        <span className="num pt-[3px] text-[11px] text-accent">
                          {String(index + 1).padStart(2, "0")}
                        </span>
                        <span>{item}</span>
                      </li>
                    ))}
                  </ol>
                </div>
              ) : null}

              {decision.key_risks?.length ? (
                <div className="mt-4 border-t border-line pt-4">
                  <Eyebrow>Key risks</Eyebrow>
                  <ul className="mt-2 space-y-1">
                    {decision.key_risks.map((risk) => (
                      <li key={risk} className="relative pl-3.5 text-[13px] text-ink-2">
                        <span className="absolute left-0 top-[0.62em] h-px w-1.5 bg-line-strong" />
                        {risk}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>

            <footer className="flex flex-wrap gap-x-6 gap-y-2 border-t border-line bg-sunken px-6 py-3 text-[11.5px] text-ink-3">
              <span>{decision.source ?? "unknown source"}</span>
              <span>
                Entry ref <span className="num">{money(decision.entry_price)}</span>
              </span>
              <span>
                Stop <span className="num">{money(decision.stop_price)}</span>
              </span>
              <span>
                Target <span className="num">{money(decision.target_price)}</span>
              </span>
              {desk.history[0]?.duration_ms ? (
                <span>Cycle {duration(desk.history[0].duration_ms)}</span>
              ) : null}
              {decision.degraded ? <span className="text-warn">decided degraded</span> : null}
            </footer>
          </>
        )}
      </article>
    </section>
  );
}

function Loading() {
  return (
    <div className="p-6">
      <Eyebrow>Stance</Eyebrow>
      <div className="mt-1.5 mb-4">
        <Skeleton w="150px" h="44px" />
      </div>
      <div className="max-w-[66ch] space-y-2.5">
        <Skeleton w="100%" h="11px" />
        <Skeleton w="94%" h="11px" />
        <Skeleton w="88%" h="11px" />
        <Skeleton w="62%" h="11px" />
      </div>
    </div>
  );
}

function Figure({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex min-w-[88px] flex-col gap-0.5">
      <dt>
        <Eyebrow>{label}</Eyebrow>
      </dt>
      <dd className="num text-[15px]">{value}</dd>
    </div>
  );
}

/** Ten segments, with the entry gate drawn where it actually falls. */
function ConfidenceSegments({ value, gate }: { value: number; gate: number | null }) {
  const filled = Math.round((Number.isFinite(value) ? value : 0) * 10);
  const gateIndex = gate === null ? -1 : Math.round(gate * 10);
  return (
    <div
      className="flex gap-[3px]"
      role="img"
      aria-label={
        gate === null
          ? `Confidence ${ratio(value)} of 1`
          : `Confidence ${ratio(value)} of 1, entry gate ${ratio(gate)}`
      }
    >
      {Array.from({ length: 10 }, (_, index) => {
        const on = index < filled;
        const isGate = index === gateIndex;
        return (
          <i
            key={index}
            className={`h-2 flex-1 rounded-r1 border transition-colors duration-[220ms] ease-desk ${
              on
                ? "border-accent bg-accent"
                : isGate
                  ? "border-dashed border-ink-3 bg-transparent"
                  : "border-line bg-sunken"
            }`}
          />
        );
      })}
    </div>
  );
}

function splitParagraphs(text: string): string[] {
  return (text || "").split(/\n{2,}|\n/).map((part) => part.trim()).filter(Boolean);
}

function titleCase(word: string): string {
  return word.charAt(0) + word.slice(1).toLowerCase();
}
