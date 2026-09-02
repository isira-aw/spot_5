/**
 * Recent cycles, scannable at a glance and expandable when a row raises a
 * question. Rows are keyed by cycle id, so a reconnect updates them in place.
 */
import { useState } from "react";

import type { CycleRow } from "../api/types";
import { clockTime, duration, money, ratio } from "../lib/format";
import { Note, SectionHead } from "./primitives";

const ACTION_CLASS: Record<string, string> = {
  BUY: "text-gain font-semibold",
  SELL: "text-loss font-semibold",
  HOLD: "text-ink-2",
};

export function History({
  rows,
  loading,
  cycleSeconds,
}: {
  rows: CycleRow[];
  loading: boolean;
  cycleSeconds: number;
}) {
  const [expanded, setExpanded] = useState<string | null>(null);

  return (
    <section aria-labelledby="history-heading">
      <SectionHead
        title="Recent cycles"
        id="history-heading"
        aside={
          rows.length ? (
            <span className="shrink-0 text-[11px] text-ink-3">last {rows.length}</span>
          ) : null
        }
      />

      {loading ? (
        <div>
          {[0, 1, 2, 3, 4].map((row) => (
            <div key={row} className="flex h-[49px] items-center border-b border-line">
              <span className="skeleton h-3 w-full rounded-r1" />
            </div>
          ))}
        </div>
      ) : rows.length === 0 ? (
        <Note title="No cycles recorded.">
          Every {Math.round(cycleSeconds / 60)} minutes the desk runs a cycle and one row
          lands here with its action, confidence and the price it saw. Rows open to show what
          each engine said and why a trade was or was not placed.
        </Note>
      ) : (
        <div
          className="overflow-x-auto"
          tabIndex={0}
          role="region"
          aria-label="Recent cycles, scrollable"
        >
          <table className="w-full min-w-[560px] border-collapse text-[13px]">
            <caption className="sr-only">
              Recent decision cycles, newest first. Each row expands for detail.
            </caption>
            <thead>
              <tr>
                <Th className="w-[96px]">Time</Th>
                <Th className="w-[74px]">Action</Th>
                <Th className="w-[64px]">Conf.</Th>
                <Th className="w-[104px]">Price</Th>
                <Th>Note</Th>
                <Th className="w-[86px]">Status</Th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const open = expanded === row.cycle_id;
                return (
                  <Row
                    key={row.cycle_id}
                    row={row}
                    open={open}
                    onToggle={() => setExpanded(open ? null : row.cycle_id)}
                  />
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function Row({
  row,
  open,
  onToggle,
}: {
  row: CycleRow;
  open: boolean;
  onToggle: () => void;
}) {
  const blocked = (row.blocked_by?.length ?? 0) > 0;
  const status = row.error ? "failed" : blocked ? "blocked" : row.status;
  const statusClass =
    row.error || blocked ? "text-warn" : status === "ok" ? "text-ink-2" : "text-ink-2";

  return (
    <>
      <tr>
        <Td className="num">{clockTime(row.started_at)}</Td>
        <Td className={ACTION_CLASS[row.action ?? ""] ?? "text-ink-3"}>{row.action ?? "—"}</Td>
        <Td className="num">{ratio(row.confidence)}</Td>
        <Td className="num">{money(row.price)}</Td>
        <Td>
          <span className="line-clamp-2">{firstSentence(row.rationale) || "—"}</span>
          <button
            type="button"
            onClick={onToggle}
            aria-expanded={open}
            aria-controls={`cycle-${row.cycle_id}`}
            className="mt-0.5 text-[12px] text-accent underline-offset-2 hover:underline"
          >
            {open ? "Hide detail" : "Detail"}
          </button>
        </Td>
        <Td className={statusClass}>{status}</Td>
      </tr>
      {open ? (
        <tr id={`cycle-${row.cycle_id}`} className="bg-sunken">
          <td colSpan={6} className="border-b border-line px-0 py-3 pr-3 text-[12.5px] text-ink-2">
            <div className="space-y-2">
              <p className="num text-[11px] text-ink-3">
                {row.cycle_id} · {duration(row.duration_ms)} · {row.source ?? "unknown source"}
              </p>
              {row.rationale ? (
                <p className="max-w-[70ch] font-prose text-[15px] leading-[1.55] text-ink">
                  {row.rationale}
                </p>
              ) : null}
              {row.blocked_by?.length ? (
                <p className="text-warn">Blocked: {row.blocked_by.join("; ")}</p>
              ) : null}
              {row.error ? <p className="text-loss">Error: {row.error}</p> : null}
            </div>
          </td>
        </tr>
      ) : null}
    </>
  );
}

function Th({ children, className = "" }: { children?: React.ReactNode; className?: string }) {
  return (
    <th
      scope="col"
      className={`border-b border-line pb-2 pr-3 text-left text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-3 ${className}`}
    >
      {children}
    </th>
  );
}

function Td({ children, className = "" }: { children?: React.ReactNode; className?: string }) {
  return (
    <td className={`border-b border-line py-2.5 pr-3 align-top text-ink-2 ${className}`}>
      {children}
    </td>
  );
}

function firstSentence(text: string | null): string {
  if (!text) return "";
  const trimmed = text.trim();
  const stop = trimmed.search(/(?<=[.!?])\s/);
  return stop > 0 ? trimmed.slice(0, stop) : trimmed;
}
