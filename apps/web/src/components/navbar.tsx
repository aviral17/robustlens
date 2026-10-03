import Link from "next/link";
import { ThemeToggle } from "./theme-toggle";

export function Navbar() {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto flex h-14 items-center justify-between px-6">
        <Link href="/" className="flex items-center gap-2 font-bold tracking-tight">
          <span className="text-xl">RobustLens</span>
        </Link>
        <nav className="flex items-center gap-6">
          <Link href="/leaderboard" className="text-sm font-medium text-muted-foreground hover:text-foreground">
            Leaderboard
          </Link>
          <Link href="/methods" className="text-sm font-medium text-muted-foreground hover:text-foreground">
            Methods
          </Link>
          <ThemeToggle />
        </nav>
      </div>
    </header>
  );
}
