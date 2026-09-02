# spot_5 — dashboard

A read-mostly view of the desk: whether it is alive, what the agent decided and
why, and what the money is doing. React + TypeScript + Vite, Tailwind for
styling, no component library and no state library — React state plus one
websocket.

```bash
cp .env.example .env.local     # point it at the API; add a token for admin controls
npm install
npm run dev                    # http://localhost:5173
npm run build                  # typecheck + production bundle into dist/
```

## Configuration

| variable | meaning |
|---|---|
| `VITE_API_BASE` | the FastAPI server, default `http://localhost:8000`. The websocket URL is derived from it. |
| `VITE_ADMIN_TOKEN` | optional. Without it the admin controls are **not rendered at all**, rather than shown as buttons that would 401. |

## How it stays honest

**The connection state is never a guess.** `live` means the socket is open;
`reconnecting` means it dropped and we are backing off (1 s, doubling, capped at
30 s, ±30 % jitter, and the cap is applied after the jitter); `offline (polling)`
means the retries have had real time to fail and REST polling of `/state` has
taken over. Data already on screen is never cleared because a connection
dropped — it is kept and marked stale. A gap in the per-connection `seq`
reconnects for a fresh snapshot rather than carrying on with state we know is
incomplete, and history is merged by `cycle_id`, so a reconnect updates rows in
place instead of appending duplicates.

**Nothing moves when a number changes.** Digits are tabular, values reserve their
space before they arrive, and the places where a tag can appear (staleness,
hints) hold their height whether it is there or not. Measured CLS over 25 s of
live updates is ~0.002, with no single shift above 0.001. Only colour animates,
over 420 ms — and not at all under `prefers-reduced-motion`.

**Loading is not a flash.** The placeholders wait 500 ms before appearing, so a
server that answers in 300 ms paints the real thing once.

**Nothing is written by accident.** Every state-changing action opens a
confirmation naming exactly what will happen; switching to REAL and resetting the
paper book also require the word typed back. No POST is ever triggered by page
load, focus or reconnect.

## Layout

```
src/
  api/          the server's shapes (types.ts) and the fetch layer (client.ts)
  live/         the websocket, its reconnection policy and the client's model
  components/   the eight regions plus a handful of primitives
  lib/          formatting (a missing value is an em dash, never NaN) and clocks
  index.css     every colour, type, space, radius and motion token, both themes
```

Colours exist only in `index.css`. Light and dark are both defined at the token
level for all three viewer states — an explicit choice, and the un-stamped
"system" default.

## Accessibility

Axe reports no violations in either theme. Text meets 4.5:1, the page is fully
keyboard navigable with visible focus, landmarks and headings are real, scrollable
regions are focusable, and the constantly-changing values sit behind one polite
live region rather than announcing every tick.
