"use client";

import { motion } from "motion/react";
import Link from "next/link";

const features = [
  {
    title: "Bootstrap CIs",
    description: "95% confidence intervals on accuracy via percentile and BCa bootstrap",
    icon: "📊",
  },
  {
    title: "Significance Tests",
    description: "McNemar's test & paired bootstrap with Holm/BH correction",
    icon: "🔬",
  },
  {
    title: "Calibration",
    description: "ECE, reliability diagrams, and temperature scaling",
    icon: "🎯",
  },
  {
    title: "Conformal Prediction",
    description: "Coverage guarantees vs prediction set size under shift",
    icon: "🛡️",
  },
  {
    title: "Ensemble Analysis",
    description: "Soft voting, weighted, and stacking — does averaging help?",
    icon: "🤝",
  },
  {
    title: "Robustness Score",
    description: "A single metric with uncertainty quantification",
    icon: "⚡",
  },
];

export default function HomePage() {
  return (
    <main className="flex min-h-screen flex-col">
      {/* Hero */}
      <section className="relative flex flex-col items-center justify-center px-6 py-24 text-center md:py-32">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
        >
          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl lg:text-7xl">
            <span className="bg-gradient-to-r from-zinc-200 via-zinc-400 to-zinc-600 bg-clip-text text-transparent dark:from-zinc-100 dark:via-zinc-300 dark:to-zinc-500">
              RobustLens
            </span>
          </h1>
          <p className="mx-auto mt-4 max-w-2xl text-lg text-muted-foreground sm:text-xl">
            How robust are modern vision models? A rigorous statistical evaluation
            under distribution shift.
          </p>
        </motion.div>

        <motion.div
          className="mt-8 flex flex-wrap justify-center gap-4"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2, ease: "easeOut" }}
        >
          <Link
            href="/leaderboard"
            className="inline-flex h-11 items-center justify-center rounded-md bg-primary px-8 text-sm font-medium text-primary-foreground ring-offset-background transition-colors hover:bg-primary/90"
          >
            View Leaderboard
          </Link>
          <Link
            href="/methods"
            className="inline-flex h-11 items-center justify-center rounded-md border border-input bg-background px-8 text-sm font-medium ring-offset-background transition-colors hover:bg-accent hover:text-accent-foreground"
          >
            Methodology
          </Link>
        </motion.div>
      </section>

      {/* Features Grid */}
      <section className="mx-auto w-full max-w-6xl px-6 py-16">
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((feature, i) => (
            <motion.div
              key={feature.title}
              className="group rounded-lg border bg-card p-6 transition-colors hover:bg-accent/50"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                duration: 0.4,
                delay: 0.3 + i * 0.08,
                ease: "easeOut",
              }}
            >
              <span className="text-2xl" role="img" aria-label={feature.title}>
                {feature.icon}
              </span>
              <h3 className="mt-3 text-lg font-semibold">{feature.title}</h3>
              <p className="mt-1 text-sm text-muted-foreground">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="mt-auto border-t py-8 text-center text-sm text-muted-foreground">
        <p>
          RobustLens — Built with PyTorch, timm, Next.js, and rigorous statistics.
        </p>
        <p className="mt-1">
          <Link href="/about" className="underline underline-offset-4 hover:text-foreground">
            About this project
          </Link>
        </p>
      </footer>
    </main>
  );
}
