/**
 * engine_2's model factory, driven from the page instead of a terminal.
 *
 * Three things an operator needs to see and act on, in this order: is the live
 * model still working (drift), what is it and what can I fall back to
 * (versions), and start a retrain. Training is hours long, so a job is started
 * in the background and this panel polls it — the buttons stay disabled while
 * one is running rather than letting a second one thrash the same directory.
 *
 * Nothing here can trade. Every control produces, promotes or rolls back a model
 * artifact; the desk's own execution controls live in Controls.
 */
import { useCallback, useEffect, useState } from "react";

import { admin, api, hasAdminToken } from "../api/client";
import type { Engine2Drift, Engine2Job, Engine2Models } from "../api/types";
import { clockTime, ratio } from "../lib/format";
import type { ConfirmRequest } from "./Confirm";
import { Button, Note, Pill, SectionHead, Skeleton } from "./primitives";

const RUNNING_POLL_MS = 4000;
const IDLE_POLL_MS = 60_000;

function driftTone(drift: Engine2Drift | undefined) {
  if (!drift || drift.error) return { tone: "quiet" as const, label: "unknown" };
  if (drift.verdict === "warming_up") return { tone: "quiet" as const, label: "warming up" };
  if (drift.verdict === "degraded") return { tone: "armed" as const, label: "degraded" };
  return { tone: "quiet" as const, label: "healthy" };
}

export function ModelFactory({
  onConfirm,
}: {
  onConfirm: (request: ConfirmRequest) => void;
}) {
  const [models, setModels] = useState<Engine2Models | null>(null);
  const [job, setJob] = useState<Engine2Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  const running = job?.state === "running";

  const load = useCallback(async () => {
    try {
      const next = await api.engine2Models();
      setModels(next);
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
    if (hasAdminToken) {
      try {
        setJob(await admin.engine2Job());
      } catch {
        /* the job endpoint is admin-only; a missing token is not an error here */
      }
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // Poll fast while a job runs, slowly when idle. No polling teardown surprises:
  // the interval is rebuilt whenever the rate should change.
  useEffect(() => {
    const id = window.setInterval(() => void load(), running ? RUNNING_POLL_MS : IDLE_POLL_MS);
    return () => window.clearInterval(id);
  }, [load, running]);

  if (error && !models) {
    return (
      <section aria-labelledby="factory-heading">
        <SectionHead title="Model factory" id="factory-heading" />
        <Note title="engine_2 is not reachable" tone="warn">
          {error}
        </Note>
      </section>
    );
  }

  if (!models) {
    return (
      <section aria-labelledby="factory-heading">
        <SectionHead title="Model factory" id="factory-heading" />
        <Skeleton w="24ch" />
      </section>
    );
  }

  if (!models.available) {
    return (
      <section aria-labelledby="factory-heading">
        <SectionHead title="Model factory" id="factory-heading" />
        <Note title="engine_2 is not installed here" tone="quiet">
          {models.error ?? "This process cannot import engine_2."} Training needs its own
          dependencies — <code>pip install -r backend/engine_2/requirements.txt</code> — and is
          usually run on a separate machine.
        </Note>
      </section>
    );
  }

  const drift = models.drift;
  const { tone: dTone, label: dLabel } = driftTone(drift);
  const current = models.current?.version;
  const previous = models.current?.previous;
  const history = models.history ?? [];

  const start = (body: { job: string; walkforward?: boolean }) => async () => {
    const next = await admin.engine2Start(body);
    setJob(next);
    return next;
  };

  return (
    <section aria-labelledby="factory-heading">
      <SectionHead
        title="Model factory"
        id="factory-heading"
        aside={<Pill tone={dTone}>{dLabel}</Pill>}
      />

      <dl className="mb-3 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-[13px]">
        <dt className="text-ink-2">Serving</dt>
        <dd className="font-medium">{current ?? "nothing promoted yet"}</dd>
        {previous ? (
          <>
            <dt className="text-ink-2">Previous</dt>
            <dd>{previous}</dd>
          </>
        ) : null}
        {drift && !drift.error && drift.verdict !== "warming_up" ? (
          <>
            <dt className="text-ink-2">Live accuracy</dt>
            <dd>
              {ratio(drift.dir_acc, 3)} over {drift.n ?? 0} matured predictions
              {drift.pred_std !== undefined ? `, spread ${ratio(drift.pred_std, 3)}` : ""}
            </dd>
          </>
        ) : null}
        {drift?.verdict === "warming_up" ? (
          <>
            <dt className="text-ink-2">Live accuracy</dt>
            <dd className="text-ink-2">
              {drift.n ?? 0} of {drift.needed ?? 0} predictions matured
            </dd>
          </>
        ) : null}
      </dl>

      {drift?.retrain_recommended ? (
        <div className="mb-3">
          <Note title="The live forecaster has decayed" tone="warn">
            {drift.reasons?.join("; ") ?? "Live accuracy has been below the floor."} A retrain is
            recommended. Promotion is still gated, so a worse model cannot replace this one.
          </Note>
        </div>
      ) : null}

      {job && job.state ? (
        <div className="mb-3 text-[13px]">
          <b className="font-semibold">
            {running ? "Running" : job.state === "gated" ? "Stopped at a gate" : job.state}
          </b>{" "}
          <span className="text-ink-2">
            {job.job}
            {job.step ? ` — ${job.step}` : ""}
            {job.detail ? `: ${job.detail}` : ""}
          </span>
          {job.state === "gated" && job.result?.reasons?.length ? (
            <div className="mt-1 text-ink-2">
              {job.result.gate}: {job.result.reasons.join("; ")}. The live model was not
              touched.
            </div>
          ) : null}
          {job.state === "failed" && job.error ? (
            <div className="mt-1 text-loss">{job.error}</div>
          ) : null}
          {job.state === "succeeded" && job.result?.promote ? (
            <div className="mt-1 text-ink-2">
              {job.result.promote.promoted
                ? `Promoted ${job.result.promote.version}.`
                : `Not promoted: ${job.result.promote.reasons?.join("; ")}. Live model untouched.`}
            </div>
          ) : null}
          {job.finished_at ? (
            <div className="mt-1 text-ink-2">finished {clockTime(job.finished_at)}</div>
          ) : null}
        </div>
      ) : null}

      {hasAdminToken ? (
        <div className="flex flex-wrap gap-2">
          <Button
            disabled={running}
            onClick={() =>
              onConfirm({
                title: "Pull fresh market data?",
                confirmLabel: "Pull data",
                tone: "primary",
                body: (
                  <>
                    Fetches the missing candles from the exchange and rebuilds the training
                    dataset. Read-only market data; no model is changed and nothing is traded.
                  </>
                ),
                run: start({ job: "pull" }),
              })
            }
          >
            Pull data
          </Button>

          <Button
            tone="primary"
            disabled={running}
            onClick={() =>
              onConfirm({
                title: "Retrain and promote?",
                confirmLabel: "Start training",
                tone: "primary",
                body: (
                  <>
                    Runs the full cycle in the background: fetch, train the forecaster, train the
                    agent, then score both on unseen data. It takes <b>hours</b>. The new model
                    replaces the live one <b>only</b> if it clears the gates and the held-out
                    backtest — otherwise {current ?? "the current model"} keeps serving.
                  </>
                ),
                run: start({ job: "cycle" }),
              })
            }
          >
            Retrain &amp; promote
          </Button>

          <Button
            disabled={running}
            onClick={() =>
              onConfirm({
                title: "Run a walk-forward check?",
                confirmLabel: "Start walk-forward",
                tone: "primary",
                body: (
                  <>
                    Retrains and tests across rolling folds to ask whether the edge holds in most
                    periods or just the latest one. Slower than a normal cycle and promotes
                    nothing — it is a measurement.
                  </>
                ),
                run: start({ job: "walkforward" }),
              })
            }
          >
            Walk-forward check
          </Button>

          <Button
            tone="danger"
            disabled={running || !previous}
            onClick={() =>
              onConfirm({
                title: `Roll back to ${previous}?`,
                confirmLabel: "Roll back",
                tone: "danger",
                body: (
                  <>
                    The desk will serve <b>{previous}</b> again from the next bar, instead of{" "}
                    <b>{current}</b>. Both bundles stay on disk, so this is reversible.
                  </>
                ),
                run: async () => {
                  const result = await admin.engine2Rollback();
                  await load();
                  return result;
                },
              })
            }
          >
            Roll back
          </Button>
        </div>
      ) : (
        <Note title="Read-only" tone="quiet">
          Set an admin token to start training or roll a model back from here.
        </Note>
      )}

      {history.length > 1 ? (
        <details className="mt-3 text-[13px]">
          <summary className="cursor-pointer text-ink-2">
            {history.length} versions kept
            {models.retention ? ` (newest ${models.retention} plus the live one)` : ""}
          </summary>
          <ul className="mt-2 space-y-1">
            {history.map((v) => (
              <li key={v.version} className="flex gap-3">
                <span className={v.current ? "font-medium" : "text-ink-2"}>{v.version}</span>
                {v.meta?.metrics?.sharpe !== undefined ? (
                  <span className="text-ink-2">sharpe {ratio(v.meta.metrics.sharpe, 2)}</span>
                ) : null}
                {v.current ? <span className="text-ink-2">— serving</span> : null}
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </section>
  );
}
