/**
 * One line, one baseline, no gridlines.
 *
 * Drawn as inline SVG rather than through a charting library: it is a single
 * series, and drawing it directly keeps every colour on the theme tokens (so it
 * follows light and dark without a re-render) and costs no dependency.
 */
import { useId, useMemo } from "react";

import type { EquityPoint } from "../api/types";
import { money, percent } from "../lib/format";
import { Note, SectionHead, Skeleton } from "./primitives";

const WIDTH = 320;
const HEIGHT = 132;
const PAD = 8;

export function EquityCurve({
  points,
  loading,
  cycleSeconds,
}: {
  points: EquityPoint[];
  loading: boolean;
  cycleSeconds: number;
}) {
  const gradientId = useId();
  const geometry = useMemo(() => buildGeometry(points), [points]);
  const first = points[0];
  const last = points[points.length - 1];
  const peak = points.length ? Math.max(...points.map((p) => p.equity)) : null;

  return (
    <section aria-labelledby="equity-heading">
      <SectionHead
        title="Equity curve"
        id="equity-heading"
        aside={
          points.length ? (
            <span className="shrink-0 text-[11px] text-ink-3">{points.length} points</span>
          ) : null
        }
      />

      {loading ? (
        <Skeleton w="100%" h="132px" />
      ) : points.length === 0 ? (
        <Note title="The curve starts with the first cycle.">
          One point is written every cycle — that is every {Math.round(cycleSeconds / 60)}{" "}
          minutes — so the line appears shortly after the desk begins and grows from there.
        </Note>
      ) : points.length === 1 ? (
        <div>
          <p className="num text-[19px]">{money(first?.equity)}</p>
          <p className="mt-1 max-w-[40ch] text-[13px] text-ink-2">
            One point so far. A second cycle gives the curve something to draw.
          </p>
        </div>
      ) : (
        <>
          <svg
            viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
            preserveAspectRatio="none"
            className="block h-[132px] w-full"
            role="img"
            aria-label={`Equity from ${money(first?.equity)} to ${money(last?.equity)} over ${points.length} cycles, peak ${money(peak)}.`}
          >
            <defs>
              <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--color-accent)" stopOpacity="0.22" />
                <stop offset="100%" stopColor="var(--color-accent)" stopOpacity="0" />
              </linearGradient>
            </defs>
            <line
              x1="0"
              y1={geometry.baseline}
              x2={WIDTH}
              y2={geometry.baseline}
              stroke="var(--color-line)"
              strokeWidth="1"
              fill="none"
            />
            <path d={geometry.fill} fill={`url(#${gradientId})`} stroke="none" />
            <path
              d={geometry.line}
              fill="none"
              stroke="var(--color-accent)"
              strokeWidth="1.6"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            <circle
              cx={geometry.last[0]}
              cy={geometry.last[1]}
              r="3"
              fill="var(--color-accent)"
            />
          </svg>
          <div className="flex justify-between pt-1.5 text-[11px] text-ink-3">
            <span className="num">{money(first?.equity)} start</span>
            <span className="num">peak {money(peak)}</span>
            <span className="num">
              now {money(last?.equity)}
              {last && last.drawdown_pct > 0 ? ` · −${percent(last.drawdown_pct)}` : ""}
            </span>
          </div>
        </>
      )}
    </section>
  );
}

function buildGeometry(points: EquityPoint[]) {
  const values = points.map((p) => p.equity).filter((v) => Number.isFinite(v));
  const min = values.length ? Math.min(...values) : 0;
  const max = values.length ? Math.max(...values) : 1;
  const flat = max - min === 0;
  const span = flat ? 1 : max - min;
  const step = values.length > 1 ? (WIDTH - PAD * 2) / (values.length - 1) : 0;

  const y = (value: number) =>
    flat ? HEIGHT / 2 : HEIGHT - PAD - ((value - min) / span) * (HEIGHT - PAD * 2);

  const coords = values.map((value, index): [number, number] => [
    PAD + index * step,
    y(value),
  ]);

  const line = coords
    .map(([x, y], index) => `${index === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`)
    .join(" ");

  return {
    line,
    fill: `${line} L${WIDTH - PAD} ${HEIGHT - PAD} L${PAD} ${HEIGHT - PAD} Z`,
    last: coords[coords.length - 1] ?? [PAD, HEIGHT - PAD],
    // The baseline sits at the opening equity, so gains and losses read against it.
    baseline: y(values[0] ?? min),
  };
}
