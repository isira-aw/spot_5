/**
 * The handful of pieces the dashboard repeats. Deliberately few and deliberately
 * plain: hairlines and space carry the structure, so almost nothing here draws a
 * box.
 */
import type { ReactNode } from "react";

export function Eyebrow({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <span
      className={`text-[10px] font-semibold uppercase tracking-[0.14em] text-ink-3 ${className}`}
    >
      {children}
    </span>
  );
}

export function SectionHead({
  title,
  id,
  meta,
  aside,
}: {
  title: string;
  id?: string;
  meta?: ReactNode;
  aside?: ReactNode;
}) {
  return (
    <div className="mb-4 flex items-baseline gap-3 border-b border-line pb-2">
      <h2 id={id}>
        <Eyebrow>{title}</Eyebrow>
      </h2>
      <span className="min-w-0 flex-1 truncate text-[11px] text-ink-4">{meta}</span>
      {aside}
    </div>
  );
}

type PillTone = "paper" | "real" | "quiet" | "armed" | "warn";

const PILL_TONES: Record<PillTone, string> = {
  paper: "bg-accent-soft text-accent-ink border-accent/30",
  real: "bg-hazard text-white border-hazard",
  quiet: "bg-transparent text-ink-3 border-line-strong",
  armed: "bg-loss-soft text-loss border-loss/40",
  warn: "bg-warn-soft text-warn border-warn/40",
};

export function Pill({
  tone = "quiet",
  children,
  title,
}: {
  tone?: PillTone;
  children: ReactNode;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={`inline-flex h-[22px] items-center gap-1.5 rounded-full border px-2.5 text-[11px] font-semibold uppercase tracking-[0.06em] ${PILL_TONES[tone]}`}
    >
      {children}
    </span>
  );
}

export function Dot({ tone }: { tone: "live" | "warn" | "off" }) {
  const colour =
    tone === "live" ? "bg-gain text-gain" : tone === "warn" ? "bg-warn text-warn" : "bg-ink-4";
  return (
    <span
      aria-hidden="true"
      className={`relative inline-block size-[7px] shrink-0 rounded-full ${colour} ${
        tone === "off" ? "" : "breathe"
      }`}
    />
  );
}

export function StaleTag({ children }: { children: ReactNode }) {
  return (
    <span className="rounded-r1 bg-warn-soft px-1.5 py-px text-[10px] font-semibold uppercase tracking-[0.08em] text-warn">
      {children}
    </span>
  );
}

/** A value placeholder that occupies exactly the space the value will. */
export function Skeleton({ w = "6ch", h = "1em" }: { w?: string; h?: string }) {
  // `h` defaults to 1em so the line box matches the text that will replace it.
  return (
    <span
      aria-hidden="true"
      className="skeleton inline-block align-middle"
      style={{ width: w, height: h }}
    >
      &nbsp;
    </span>
  );
}

/** Empty, error and stale copy. Never just "No data". */
export function Note({
  title,
  tone = "quiet",
  children,
  action,
}: {
  title: string;
  tone?: "quiet" | "warn" | "error";
  children: ReactNode;
  action?: ReactNode;
}) {
  const heading =
    tone === "error" ? "text-loss" : tone === "warn" ? "text-warn" : "text-ink";
  return (
    <div className="max-w-[42ch] text-[13px] text-ink-2">
      <b className={`mb-1 block font-semibold ${heading}`}>{title}</b>
      {children}
      {action ? <div className="mt-3">{action}</div> : null}
    </div>
  );
}

export function Button({
  children,
  onClick,
  tone = "plain",
  disabled,
  type = "button",
  ...rest
}: {
  children: ReactNode;
  onClick?: () => void;
  tone?: "plain" | "danger" | "primary";
  disabled?: boolean;
  type?: "button" | "submit";
} & Record<string, unknown>) {
  const tones = {
    plain: "border-line-strong text-ink hover:bg-sunken hover:border-ink-4",
    danger: "border-loss/45 text-loss hover:bg-loss-soft",
    primary: "border-accent bg-accent text-white hover:bg-accent-ink",
  } as const;
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex h-7 items-center gap-1.5 rounded-r2 border bg-surface px-3 text-[12px] font-medium transition-colors duration-[120ms] disabled:cursor-not-allowed disabled:opacity-45 ${tones[tone]}`}
      {...rest}
    >
      {children}
    </button>
  );
}

/** A horizontal meter. Width is the only thing that animates. */
export function Meter({
  value,
  tone = "accent",
  label,
}: {
  value: number;
  tone?: "accent" | "gain" | "loss" | "neutral" | "warn";
  label?: string;
}) {
  const fill = {
    accent: "bg-accent",
    gain: "bg-gain",
    loss: "bg-loss",
    warn: "bg-warn",
    neutral: "bg-ink-2",
  }[tone];
  return (
    <div
      className="h-1 overflow-hidden rounded-full bg-sunken"
      role={label ? "img" : undefined}
      aria-label={label}
    >
      <span
        className={`block h-full transition-[width] duration-[420ms] ease-desk ${fill}`}
        style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }}
      />
    </div>
  );
}
