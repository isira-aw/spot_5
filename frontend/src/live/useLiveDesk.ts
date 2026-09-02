/**
 * One websocket, one fallback, one truthful connection state.
 *
 *   live          the socket is open and the desk is talking to us
 *   reconnecting  the socket dropped; we are backing off and retrying
 *   offline       retries have not landed; we are polling /state instead
 *
 * The socket is the fast path and REST is the floor: if the socket never
 * connects at all the dashboard still works, just at 10-second resolution. Data
 * already on screen is never cleared because a connection dropped — it is kept
 * and marked stale.
 */
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";

import { api, websocketUrl } from "../api/client";
import type { Envelope, Snapshot } from "../api/types";
import { emptyDesk, reduceDesk, type Desk } from "./reducer";

export type Connection = "live" | "reconnecting" | "offline";

const BACKOFF_MIN_MS = 1_000;
const BACKOFF_MAX_MS = 30_000;
const OFFLINE_AFTER_ATTEMPTS = 3; // ~1s + 2s + 4s of trying before we say offline
const OFFLINE_AFTER_MS = 6_000; // ... and never sooner than this, however fast they fail
const POLL_MS = 10_000;

export interface Live {
  desk: Desk;
  connection: Connection;
  /** Populated when we could not reach the server at all. */
  error: string | null;
  attempt: number;
  /** Seconds until the next reconnect attempt, for the status bar. */
  retryInSeconds: number | null;
  /** True when a seq gap was seen — the socket reconnects to resynchronise. */
  refresh: () => void;
}

function backoffMs(attempt: number): number {
  const flat = Math.min(BACKOFF_MAX_MS, BACKOFF_MIN_MS * 2 ** Math.max(0, attempt - 1));
  const jittered = flat * (0.7 + Math.random() * 0.6); // ±30%
  return Math.round(Math.min(BACKOFF_MAX_MS, jittered)); // the cap is a real cap
}

export function useLiveDesk(): Live {
  const [desk, dispatch] = useReducer(reduceDesk, emptyDesk);
  const [connection, setConnection] = useState<Connection>("reconnecting");
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [retryAt, setRetryAt] = useState<number | null>(null);
  const [retryInSeconds, setRetryInSeconds] = useState<number | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  const droppedAtRef = useRef<number | null>(null);
  const attemptRef = useRef(0);
  const timerRef = useRef<number | null>(null);
  const pollRef = useRef<number | null>(null);
  const seqRef = useRef(0);
  const closedRef = useRef(false);
  const connectRef = useRef<() => void>(() => {});

  // ── REST fallback ─────────────────────────────────────────────────────────
  const pollOnce = useCallback(async () => {
    try {
      const [state, health] = await Promise.all([api.state(), api.health()]);
      dispatch({
        kind: "state",
        data: { ...state, health } as Snapshot,
        at: Date.now(),
      });
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The server is not answering.");
    }
  }, []);

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const startPolling = useCallback(() => {
    if (pollRef.current !== null) return;
    void pollOnce();
    pollRef.current = window.setInterval(() => void pollOnce(), POLL_MS);
    // Backfill the history and curve the socket would have sent in a snapshot.
    void api
      .decisions(20)
      .then((rows) => dispatch({ kind: "history", data: rows, at: Date.now() }))
      .catch(() => undefined);
    void api
      .equity(200)
      .then((rows) => dispatch({ kind: "equity", data: rows, at: Date.now() }))
      .catch(() => undefined);
  }, [pollOnce]);

  // ── the socket ────────────────────────────────────────────────────────────
  const scheduleReconnect = useCallback(() => {
    if (closedRef.current) return;
    if (droppedAtRef.current === null) droppedAtRef.current = Date.now();
    const next = (attemptRef.current += 1);
    const wait = backoffMs(next);
    // Offline is a claim about the server, so make it only once the retries have
    // actually had time to fail — not merely because they failed fast.
    const givenUp =
      next >= OFFLINE_AFTER_ATTEMPTS && Date.now() - droppedAtRef.current >= OFFLINE_AFTER_MS;
    setAttempt(next);
    setConnection(givenUp ? "offline" : "reconnecting");
    if (givenUp) startPolling();
    setRetryAt(Date.now() + wait);
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(() => connectRef.current(), wait);
  }, [startPolling]);

  const connect = useCallback(() => {
    if (closedRef.current) return;
    let socket: WebSocket;
    try {
      socket = new WebSocket(websocketUrl());
    } catch {
      scheduleReconnect();
      return;
    }
    socketRef.current = socket;

    socket.onopen = () => {
      seqRef.current = 0;
      droppedAtRef.current = null;
      attemptRef.current = 0;
      setAttempt(0);
      setRetryAt(null);
      setConnection("live");
      setError(null);
      stopPolling(); // the socket is authoritative again
    };

    socket.onmessage = (raw) => {
      let frame: Envelope;
      try {
        frame = JSON.parse(raw.data as string) as Envelope;
      } catch {
        return;
      }
      const at = Date.now();

      // A gap means we missed something. Reconnect for a fresh snapshot rather
      // than carrying on with a state we know is incomplete.
      if (frame.seq !== seqRef.current + 1 && frame.type !== "snapshot") {
        if (frame.seq > seqRef.current + 1) {
          socket.close(4000, "sequence gap");
          return;
        }
      }
      seqRef.current = frame.seq;

      switch (frame.type) {
        case "ping":
          socket.send(JSON.stringify({ type: "pong" }));
          return;
        case "snapshot":
          dispatch({ kind: "snapshot", data: frame.data as Snapshot, at });
          return;
        case "price":
          dispatch({ kind: "price", data: frame.data as never, at });
          return;
        case "portfolio":
          dispatch({ kind: "portfolio", data: frame.data as never, at });
          return;
        case "decision":
          dispatch({ kind: "decision", data: frame.data as never, at });
          return;
        case "cycle_start":
          dispatch({ kind: "cycle_start", data: frame.data as never, at });
          return;
        case "trade":
          dispatch({ kind: "trade", data: frame.data as never, at, ts: frame.ts });
          return;
        case "event":
          dispatch({ kind: "event", data: frame.data as never, at });
          return;
        case "health":
          dispatch({ kind: "health", data: frame.data as never, at });
          return;
        default:
          return;
      }
    };

    socket.onerror = () => {
      // onclose always follows; the reconnect is scheduled there.
    };

    socket.onclose = () => {
      if (socketRef.current === socket) socketRef.current = null;
      if (closedRef.current) return;
      setConnection((current) => (current === "offline" ? "offline" : "reconnecting"));
      scheduleReconnect();
    };
  }, [scheduleReconnect, stopPolling]);

  connectRef.current = connect;

  useEffect(() => {
    closedRef.current = false;
    connect();
    return () => {
      closedRef.current = true;
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
      stopPolling();
      socketRef.current?.close();
      socketRef.current = null;
    };
    // Connect once, on mount. Reconnection is driven by onclose, never by React.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Countdown to the next attempt, so the status bar can be specific.
  useEffect(() => {
    if (retryAt === null) {
      setRetryInSeconds(null);
      return;
    }
    const tick = () => setRetryInSeconds(Math.max(0, Math.ceil((retryAt - Date.now()) / 1000)));
    tick();
    const id = window.setInterval(tick, 1000);
    return () => window.clearInterval(id);
  }, [retryAt]);

  /** A manual retry: closes the socket so the normal reconnect path runs. */
  const refresh = useCallback(() => {
    if (connection === "live") {
      void pollOnce();
      return;
    }
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    connectRef.current();
  }, [connection, pollOnce]);

  return useMemo(
    () => ({ desk, connection, error, attempt, retryInSeconds, refresh }),
    [desk, connection, error, attempt, retryInSeconds, refresh],
  );
}
