/**
 * Nothing that changes the desk happens on a single click.
 *
 * Every action names exactly what will happen. The destructive ones — switching
 * to REAL money, wiping the paper book — also require the phrase typed back, so
 * a mis-click cannot reach the broker.
 */
import { useEffect, useId, useRef, useState } from "react";

import { Button } from "./primitives";

export interface ConfirmRequest {
  title: string;
  body: React.ReactNode;
  confirmLabel: string;
  /** When set, the user must type this exact word before confirming. */
  typed?: string;
  tone?: "danger" | "primary";
  run: () => Promise<unknown>;
}

export function ConfirmDialog({
  request,
  onClose,
}: {
  request: ConfirmRequest | null;
  onClose: () => void;
}) {
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const titleId = useId();
  const bodyId = useId();

  useEffect(() => {
    setTyped("");
    setError(null);
    setBusy(false);
    if (request) {
      const target = dialogRef.current?.querySelector<HTMLElement>(
        "input, button:not([disabled])",
      );
      target?.focus();
    }
  }, [request]);

  useEffect(() => {
    if (!request) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
      if (event.key !== "Tab" || !dialogRef.current) return;
      const focusable = dialogRef.current.querySelectorAll<HTMLElement>(
        "button:not([disabled]), input, [href], [tabindex]:not([tabindex='-1'])",
      );
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (!first || !last) return;
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [request, onClose]);

  if (!request) return null;

  const ready = !request.typed || typed.trim().toUpperCase() === request.typed.toUpperCase();

  const submit = async () => {
    if (!ready || busy) return;
    setBusy(true);
    setError(null);
    try {
      await request.run();
      onClose();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The server refused that.");
      setBusy(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink/40 p-4"
      onMouseDown={(event) => event.target === event.currentTarget && onClose()}
    >
      <div
        ref={dialogRef}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={bodyId}
        className="w-full max-w-[440px] rounded-r3 border border-line bg-surface p-6 shadow-lift"
      >
        <h2 id={titleId} className="font-prose text-[22px] font-medium tracking-[-0.01em]">
          {request.title}
        </h2>
        <div id={bodyId} className="mt-3 text-[13.5px] leading-relaxed text-ink-2">
          {request.body}
        </div>

        {request.typed ? (
          <label className="mt-4 block text-[13px] text-ink-2">
            Type <b className="num text-ink">{request.typed}</b> to confirm
            <input
              value={typed}
              onChange={(event) => setTyped(event.target.value)}
              onKeyDown={(event) => event.key === "Enter" && void submit()}
              autoComplete="off"
              spellCheck={false}
              className="num mt-1.5 block w-full rounded-r2 border border-line-strong bg-bg px-3 py-2 text-[14px] text-ink outline-none focus-visible:border-accent"
            />
          </label>
        ) : null}

        {error ? (
          <p role="alert" className="mt-3 text-[13px] text-loss">
            {error}
          </p>
        ) : null}

        <div className="mt-5 flex justify-end gap-2">
          <Button onClick={onClose}>Cancel</Button>
          <Button
            tone={request.tone ?? "danger"}
            onClick={() => void submit()}
            disabled={!ready || busy}
          >
            {busy ? "Working…" : request.confirmLabel}
          </Button>
        </div>
      </div>
    </div>
  );
}
