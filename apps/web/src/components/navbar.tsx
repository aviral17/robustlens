"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "./theme-toggle";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { Menu, X } from "lucide-react";

const navLinks = [
  { href: "/", label: "Home" },
  { href: "/leaderboard", label: "Leaderboard" },
  { href: "/order-effects", label: "Order Effects" },
  { href: "/explorer", label: "Explorer" },
  { href: "/significance", label: "Significance" },
  { href: "/calibration", label: "Calibration" },
  { href: "/conformal", label: "Conformal" },
  { href: "/methods", label: "Methods" },
  { href: "/about", label: "About" },
];

export function Navbar() {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <div className="fixed top-0 inset-x-0 z-50 flex justify-center px-2.5 sm:px-6 pt-2.5 sm:pt-4 pointer-events-none">
      <header
        className={cn(
          "w-full max-w-[1400px] border border-border/50 bg-background/85 px-3 sm:px-5 py-2 sm:py-2.5 shadow-md backdrop-blur-xl dark:shadow-none supports-[backdrop-filter]:bg-background/70 pointer-events-auto transition-all duration-200",
          mobileMenuOpen ? "rounded-3xl" : "rounded-full"
        )}
      >
        <div className="flex items-center justify-between w-full">
          {/* Brand / Logo: Dedicated non-collapsing layout area with guaranteed gutter */}
          <div className="flex items-center shrink-0 mr-2 sm:mr-4 xl:mr-6">
            <Link
              href="/"
              className="group flex items-center gap-2 sm:gap-2.5 rounded-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            >
              <div className="flex h-8 w-8 sm:h-9 sm:w-9 xl:h-10 xl:w-10 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-sm transition-transform group-hover:scale-105 shrink-0">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="h-4 w-4 sm:h-4.5 sm:w-4.5 xl:h-5 xl:w-5"
                >
                  <path d="M2 12A10 10 0 0 0 22 12" />
                  <path d="M12 2v20" />
                  <path d="M12 12c-4.418 0-8-1.79-8-4" />
                  <path d="M12 12c4.418 0 8 1.79 8 4" />
                </svg>
              </div>
              <span className="bg-gradient-to-r from-foreground to-foreground/80 bg-clip-text text-base sm:text-lg xl:text-xl font-bold tracking-tight text-transparent transition-colors font-[family-name:var(--font-poppins)] whitespace-nowrap">
                RobustLens
              </span>
            </Link>
          </div>

          {/* Desktop Navigation Links (shown at xl screens >= 1280px with full breathing room) */}
          <nav className="hidden xl:flex items-center justify-center gap-1 2xl:gap-1.5 flex-1 min-w-0">
            {navLinks.map((link) => {
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={cn(
                    "relative px-3 2xl:px-4 py-1.5 text-sm font-medium transition-all duration-150 rounded-full whitespace-nowrap",
                    isActive
                      ? "text-primary-foreground bg-primary font-semibold shadow-sm"
                      : "text-muted-foreground hover:bg-muted/80 hover:text-foreground"
                  )}
                >
                  {link.label}
                </Link>
              );
            })}
          </nav>

          {/* Desktop Theme Control: Placed neatly inside the rounded container on the right */}
          <div className="hidden xl:flex items-center shrink-0 ml-3 2xl:ml-6 pl-3 2xl:pl-4 border-l border-border/60">
            <ThemeToggle />
          </div>

          {/* Mobile & Tablet Actions: Theme control + Hamburger toggle inside header */}
          <div className="flex xl:hidden items-center gap-1.5 sm:gap-2.5 shrink-0">
            <ThemeToggle />
            <button
              type="button"
              className="flex h-8 w-8 sm:h-9 sm:w-9 items-center justify-center rounded-full text-foreground hover:bg-muted transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary shrink-0"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              aria-label={mobileMenuOpen ? "Close menu" : "Open menu"}
              aria-expanded={mobileMenuOpen}
            >
              {mobileMenuOpen ? <X size={18} /> : <Menu size={18} />}
            </button>
          </div>
        </div>

        {/* Mobile & Tablet Dropdown Menu */}
        {mobileMenuOpen && (
          <nav className="xl:hidden mt-3 pt-3 border-t border-border/50 flex flex-col gap-1 pb-1 max-h-[calc(100vh-120px)] overflow-y-auto">
            {navLinks.map((link) => {
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  onClick={() => setMobileMenuOpen(false)}
                  className={cn(
                    "px-4 py-2 text-sm sm:text-base font-medium transition-colors rounded-xl",
                    isActive
                      ? "text-primary-foreground bg-primary font-semibold shadow-sm"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground"
                  )}
                >
                  {link.label}
                </Link>
              );
            })}
          </nav>
        )}
      </header>
    </div>
  );
}
