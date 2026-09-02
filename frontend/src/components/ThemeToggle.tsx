/** Light, dark, or whatever the machine says. The choice is remembered. */
import { useEffect, useState } from "react";

type Theme = "light" | "dark" | "system";
const KEY = "spot5-theme";

function read(): Theme {
  try {
    const stored = window.localStorage.getItem(KEY);
    if (stored === "light" || stored === "dark" || stored === "system") return stored;
  } catch {
    /* private mode, blocked storage — the system default is fine */
  }
  return "system";
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(read);

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try {
      window.localStorage.setItem(KEY, theme);
    } catch {
      /* nothing to do; the page still renders in the chosen theme */
    }
  }, [theme]);

  return (
    <div
      role="group"
      aria-label="Colour theme"
      className="inline-flex overflow-hidden rounded-full border border-line-strong"
    >
      {(["light", "dark", "system"] as const).map((option) => (
        <button
          key={option}
          type="button"
          aria-pressed={theme === option}
          onClick={() => setTheme(option)}
          className={`px-2.5 py-[3px] text-[11px] font-semibold capitalize transition-colors duration-[120ms] ${
            theme === option ? "bg-ink text-bg" : "text-ink-3 hover:text-ink"
          }`}
        >
          {option}
        </button>
      ))}
    </div>
  );
}
