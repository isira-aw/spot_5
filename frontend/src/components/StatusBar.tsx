/**
 * The first line of the page and the first question it answers: is the desk
 * alive? Connection, book, kill switch, next cycle, price and how old it is —
 * one row that never moves and never lies.
 */
import type { Connection } from "../live/useLiveDesk";
import type { Desk } from "../live/reducer";
import { DASH, clockTime, countdown, money, relativeAge } from "../lib/format";
import { Dot, Pill, Skeleton, StaleTag } from "./primitives";

const CONNECTION_COPY: Record<Connection, { label: string; tone: "live" | "warn" | "off" }> = {
  live: { label: "Live", tone: "live" },
  reconnecting: { label: "Reconnecting", tone: "warn" },
  offline: { label: "Offline (polling)", tone: "off" },
};

export function StatusBar({
  desk,
  connection,
  retryInSeconds,
  nextCycleSeconds,
  priceAgeSeconds,
  priceStale,
  themeControl,
}: {
  desk: Desk;
  connection: Connection;
  retryInSeconds: number | null;
  nextCycleSeconds: number | null;
  priceAgeSeconds: number | null;
  priceStale: boolean;
  themeControl: React.ReactNode;
}) {
  const { label, tone } = CONNECTION_COPY[connection];
  const real = desk.mode === "REAL";
  const killed = desk.restrictions?.kill_switch ?? desk.health?.kill_switch ?? false;
  const price = desk.price;

  return (
    <header className="sticky top-0 z-20 border-b border-line bg-rail/95 backdrop-blur">
      {real ? <div className="hazard-band h-[3px]" aria-hidden="true" /> : null}
      <div className="mx-auto flex h-[46px] max-w-[1320px] items-center px-4 sm:px-6">
        <div
          className="flex min-w-0 flex-1 items-center gap-5 overflow-x-auto [scrollbar-width:none]"
          tabIndex={0}
          role="group"
          aria-label="Desk status"
        >
        <span className="flex shrink-0 items-center gap-2 whitespace-nowrap text-[12px]">
          <Dot tone={tone} />
          <b className="font-semibold text-ink">{label}</b>
          {connection !== "live" && retryInSeconds !== null ? (
            <span className="text-ink-3">
              retrying in <span className="num">{retryInSeconds}</span> s
            </span>
          ) : null}
        </span>

        <Divider />

        <span className="flex shrink-0 items-center gap-2 whitespace-nowrap text-[12px]">
          {desk.mode ? (
            <Pill tone={real ? "real" : "paper"}>{real ? "Real money" : "Paper"}</Pill>
          ) : (
            <Skeleton w="72px" h="22px" />
          )}
          <span className="text-ink-3">{real ? "live funds" : "simulated book"}</span>
        </span>

        <Divider />

        <span className="shrink-0 whitespace-nowrap text-[12px]">
          <Pill tone={killed ? "armed" : "quiet"}>
            {killed ? "Kill switch armed" : "Kill switch off"}
          </Pill>
        </span>

        <Divider />

        <span className="flex shrink-0 items-center gap-2 whitespace-nowrap text-[12px] text-ink-2">
          {desk.runningCycleId ? (
            <>
              <span className="text-ink-3">Cycle</span>
              <b className="font-semibold text-ink">running…</b>
            </>
          ) : desk.health && !desk.health.scheduler.running ? (
            <>
              <span className="text-ink-3">Cycles</span>
              <b className="font-semibold text-warn">scheduler stopped</b>
            </>
          ) : (
            <>
              <span className="text-ink-3">Next cycle</span>
              <b className="num inline-block min-w-[5ch] text-right font-semibold text-ink">
                {countdown(nextCycleSeconds)}
              </b>
            </>
          )}
        </span>

        <Divider />

        <span className="flex shrink-0 items-center gap-2 whitespace-nowrap text-[12px] text-ink-3">
          <span className="text-ink-2">{desk.symbol || "—"}</span>
          {price === null ? (
            <Skeleton w="10ch" />
          ) : (
            <b
              className={`num inline-block w-[10ch] text-right text-[13px] font-semibold transition-colors duration-[420ms] ease-desk ${
                priceStale || !price.ok ? "text-ink-3" : "text-ink"
              }`}
            >
              {price.ok ? money(price.price) : DASH}
            </b>
          )}
          {price?.ok ? (
            <span className="inline-block w-[17ch] truncate">
              {price.source} · {relativeAge(priceAgeSeconds)}
            </span>
          ) : price ? (
            <span
              className="max-w-[26ch] truncate text-warn"
              title={price.error ?? undefined}
            >
              no price source
            </span>
          ) : null}
          {price?.ok && priceStale ? <StaleTag>stale</StaleTag> : null}
        </span>

        </div>

        <span className="flex shrink-0 items-center gap-3 pl-4">
          {/* Fixed width and tabular digits: this text changes every second and
              must never move the controls beside it. */}
          <span className="num hidden w-[15ch] text-right text-[11px] text-ink-3 lg:inline">
            {connection === "offline"
              ? "polling /state"
              : connection === "reconnecting"
                ? "reconnecting"
                : desk.updatedAt
                  ? clockTime(new Date(desk.updatedAt).toISOString())
                  : ""}
          </span>
          {themeControl}
        </span>
      </div>
    </header>
  );
}

function Divider() {
  return <span aria-hidden="true" className="h-4 w-px shrink-0 bg-line-strong" />;
}
