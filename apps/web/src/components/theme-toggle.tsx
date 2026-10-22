"use client";

import { Moon, Sun, Monitor } from "lucide-react";
import { useTheme } from "@/components/theme-provider";
import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // Prevent hydration mismatch by only rendering after mount
  useEffect(() => setMounted(true), []);
  if (!mounted) return <div className="h-9 w-24 rounded-full border bg-muted/30" />;

  const modes = [
    { key: "light" as const, icon: Sun, label: "Light" },
    { key: "dark" as const, icon: Moon, label: "Dark" },
    { key: "system" as const, icon: Monitor, label: "System" },
  ];

  return (
    <div className="flex items-center gap-0.5 sm:gap-1 rounded-full border border-border/50 bg-muted/30 p-0.5 sm:p-1 backdrop-blur-sm shrink-0">
      {modes.map(({ key, icon: Icon, label }) => (
        <button
          key={key}
          onClick={() => setTheme(key)}
          className={cn(
            "relative inline-flex h-6 w-6 sm:h-7 sm:w-7 cursor-pointer items-center justify-center rounded-full transition-all duration-300 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary",
            theme === key
              ? "bg-background text-foreground shadow-sm ring-1 ring-border"
              : "text-muted-foreground hover:bg-muted/80 hover:text-foreground"
          )}
          aria-label={`Switch to ${label} mode`}
          title={label}
        >
          <Icon className={cn("h-3 w-3 sm:h-3.5 sm:w-3.5 transition-transform", theme === key ? "scale-100" : "scale-90")} />
        </button>
      ))}
    </div>
  );
}
