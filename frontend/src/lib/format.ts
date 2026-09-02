/**
 * Formatting. Every number that reaches the screen goes through here, so a
 * missing value renders as an em dash and never as `NaN` or `undefined`.
 */
const DASH = "—";

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function money(value: unknown, digits = 2): string {
  if (!finite(value)) return DASH;
  return value.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function signedMoney(value: unknown, digits = 2): string {
  if (!finite(value)) return DASH;
  const body = money(Math.abs(value), digits);
  if (value > 0) return `+${body}`;
  if (value < 0) return `−${body}`;
  return body;
}

export function percent(value: unknown, digits = 2): string {
  if (!finite(value)) return DASH;
  return `${value.toFixed(digits)} %`;
}

export function ratio(value: unknown, digits = 2): string {
  return finite(value) ? value.toFixed(digits) : DASH;
}

export function qty(value: unknown, digits = 6): string {
  if (!finite(value)) return DASH;
  return value.toLocaleString("en-US", { maximumFractionDigits: digits });
}

export function clockTime(iso: string | null | undefined): string {
  if (!iso) return DASH;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return DASH;
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

export function countdown(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds) || seconds < 0) return "--:--";
  const whole = Math.floor(seconds);
  const m = Math.floor(whole / 60);
  const s = whole % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export function relativeAge(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) return DASH;
  if (seconds < 1) return "just now";
  if (seconds < 60) return `${Math.round(seconds)} s ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  return `${hours} h ago`;
}

export function duration(ms: number | null | undefined): string {
  if (!finite(ms)) return DASH;
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

export { DASH };
