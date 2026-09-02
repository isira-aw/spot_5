import { useEffect, useState } from "react";

/** A once-a-second clock, so ages and countdowns stay honest without polling. */
export function useNow(intervalMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(id);
  }, [intervalMs]);
  return now;
}

/**
 * True once `delayMs` has passed since mount. Used to hold the loading
 * placeholders back: a server that answers in 300 ms should paint the real
 * thing once, rather than flashing a skeleton of a different size and shoving
 * the page down when the data lands.
 */
export function useElapsed(delayMs: number): boolean {
  const [elapsed, setElapsed] = useState(false);
  useEffect(() => {
    const id = window.setTimeout(() => setElapsed(true), delayMs);
    return () => window.clearTimeout(id);
  }, [delayMs]);
  return elapsed;
}
