import Link from "next/link";
import { ThemeToggle } from "./theme-toggle";

export function Navbar() {
  return (
    <div className="fixed top-4 inset-x-0 z-50 flex justify-center px-4">
      <header className="w-full max-w-5xl rounded-full border border-border/40 bg-background/60 px-6 py-3 shadow-lg shadow-black/5 backdrop-blur-md dark:shadow-black/20 supports-[backdrop-filter]:bg-background/40">
        <div className="flex items-center justify-between">
          <Link href="/" className="group flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-sm transition-transform group-hover:scale-105">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-4 w-4"
              >
                <path d="M2 12A10 10 0 0 0 22 12" />
                <path d="M12 2v20" />
                <path d="M12 12c-4.418 0-8-1.79-8-4" />
                <path d="M12 12c4.418 0 8 1.79 8 4" />
              </svg>
            </div>
            <span className="bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-xl font-bold tracking-tight text-transparent transition-colors">
              RobustLens
            </span>
          </Link>
          <nav className="flex items-center gap-6">
            <Link
              href="/leaderboard"
              className="text-sm font-semibold text-muted-foreground transition-colors hover:text-foreground"
            >
              Leaderboard
            </Link>
            <Link
              href="/methods"
              className="text-sm font-semibold text-muted-foreground transition-colors hover:text-foreground"
            >
              Methods
            </Link>
            <div className="h-4 w-px bg-border/50"></div>
            <ThemeToggle />
          </nav>
        </div>
      </header>
    </div>
  );
}
