/**
 * The operator's controls. Present only when an admin token is configured — no
 * token, no section, rather than buttons that would come back 401.
 *
 * The kill switch is the most important control on the page, so it is first,
 * always reachable, and always states what it currently is.
 */
import { admin } from "../api/client";
import type { Desk } from "../live/reducer";
import type { ConfirmRequest } from "./Confirm";
import { Button, Pill, SectionHead } from "./primitives";

export function Controls({
  desk,
  onConfirm,
  onDone,
}: {
  desk: Desk;
  onConfirm: (request: ConfirmRequest) => void;
  onDone: () => void;
}) {
  const killed = desk.restrictions?.kill_switch ?? desk.health?.kill_switch ?? false;
  const real = desk.mode === "REAL";
  const schedulerRunning = desk.health?.scheduler.running ?? false;
  const after = async <T,>(promise: Promise<T>) => {
    const result = await promise;
    onDone();
    return result;
  };

  return (
    <section aria-labelledby="controls-heading">
      <SectionHead
        title="Controls"
        id="controls-heading"
        aside={<Pill tone={killed ? "armed" : "quiet"}>{killed ? "Armed" : "Not armed"}</Pill>}
      />

      <div className="flex flex-wrap gap-2">
        <Button
          tone={killed ? "plain" : "danger"}
          onClick={() =>
            onConfirm({
              title: killed ? "Disarm the kill switch?" : "Arm the kill switch?",
              tone: killed ? "primary" : "danger",
              confirmLabel: killed ? "Disarm" : "Arm kill switch",
              body: killed ? (
                <>
                  The desk will be allowed to open new positions again from the next cycle.
                  Existing caps and restrictions still apply.
                </>
              ) : (
                <>
                  No new position will be opened from now on, in{" "}
                  <b>{desk.mode ?? "this"}</b> mode. Positions already open stay open and their
                  stops and targets still fire — this blocks entries, it does not flatten the
                  book.
                </>
              ),
              run: () => after(admin.killSwitch(!killed, "from the dashboard")),
            })
          }
        >
          {killed ? "Disarm kill switch" : "Arm kill switch"}
        </Button>

        <Button
          onClick={() =>
            onConfirm({
              title: "Run one cycle now?",
              confirmLabel: "Run a cycle",
              tone: "primary",
              body: (
                <>
                  The desk will consult all three engines and the agent immediately, out of
                  schedule, and <b>will place an order if the agent decides to</b> — in{" "}
                  <b>{desk.mode ?? "the current"}</b> mode, inside the active caps.
                </>
              ),
              run: () => after(admin.runCycle()),
            })
          }
        >
          Run cycle now
        </Button>

        <Button
          onClick={() =>
            onConfirm({
              title: schedulerRunning ? "Stop the scheduler?" : "Start the scheduler?",
              confirmLabel: schedulerRunning ? "Stop it" : "Start it",
              tone: schedulerRunning ? "danger" : "primary",
              body: schedulerRunning ? (
                <>
                  The background loops stop: no more decision cycles, no retraining, no
                  maintenance. Open positions are left exactly as they are, unmanaged, until it
                  is started again.
                </>
              ) : (
                <>
                  The background loops start and the first decision cycle runs immediately.
                </>
              ),
              run: () => after(admin.scheduler(schedulerRunning ? "stop" : "start")),
            })
          }
        >
          {schedulerRunning ? "Stop scheduler" : "Start scheduler"}
        </Button>

        <Button
          tone={real ? "plain" : "danger"}
          onClick={() =>
            onConfirm({
              title: real ? "Switch back to PAPER?" : "Switch to REAL money?",
              confirmLabel: real ? "Switch to PAPER" : "Switch to REAL",
              typed: real ? undefined : "REAL",
              tone: real ? "primary" : "danger",
              body: real ? (
                <>
                  Trading returns to the simulated book. The REAL book is left untouched and
                  any open REAL position stays open and unmanaged until you switch back.
                </>
              ) : (
                <>
                  Every decision from the next cycle will be executed with{" "}
                  <b>actual funds</b> on the live venue. The paper book is left untouched. The
                  server runs its own preflight and will refuse if the venue is not properly
                  configured.
                </>
              ),
              run: () => after(admin.setMode(real ? "PAPER" : "REAL")),
            })
          }
        >
          {real ? "Switch to PAPER" : "Switch to REAL"}
        </Button>

        <Button
          tone="danger"
          disabled={real}
          title={real ? "Only available while the desk is in PAPER mode." : undefined}
          onClick={() =>
            onConfirm({
              title: "Reset the paper book?",
              confirmLabel: "Reset the book",
              typed: "RESET",
              body: (
                <>
                  Every simulated trade, order, position and equity point is deleted and the
                  cash is set back to the configured starting balance. Decisions and the audit
                  trail are kept. <b>The REAL book is never touched.</b> This cannot be undone.
                </>
              ),
              run: () => after(admin.resetPaper()),
            })
          }
        >
          Reset paper book
        </Button>
      </div>

      <p className="mt-3 max-w-[46ch] text-[12px] text-ink-3">
        Every control here confirms first and names exactly what it will do. Switching to REAL
        and resetting the book need the word typed back.
      </p>
    </section>
  );
}
