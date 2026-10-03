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
    <main className="flex min-h-screen flex-col items-center">
      {/* Decorative Background Glow */}
      <div className="absolute inset-0 -z-10 h-full w-full bg-background bg-[radial-gradient(#e5e7eb_1px,transparent_1px)] [background-size:16px_16px] dark:bg-[radial-gradient(#1f2937_1px,transparent_1px)] [mask-image:radial-gradient(ellipse_50%_50%_at_50%_50%,#000_70%,transparent_100%)]"></div>
      
      {/* Hero */}
      <section className="relative flex w-full flex-col items-center justify-center px-6 pt-32 pb-16 text-center md:pt-48 md:pb-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: "easeOut" }}
          className="max-w-4xl"
        >
          <div className="mb-6 flex items-center justify-center gap-2">
            <span className="rounded-full bg-blue-500/10 px-3 py-1 text-sm font-semibold text-blue-600 dark:bg-blue-400/10 dark:text-blue-400">
              Research & Evaluation Framework
            </span>
          </div>
          
          <h1 className="text-5xl font-extrabold tracking-tight sm:text-6xl md:text-7xl lg:text-8xl">
            <span className="bg-gradient-to-br from-gray-900 via-gray-700 to-gray-500 bg-clip-text text-transparent dark:from-white dark:via-gray-200 dark:to-gray-500">
              See How Models
            </span>
            <br />
            <span className="bg-gradient-to-r from-blue-600 to-violet-600 bg-clip-text text-transparent dark:from-blue-400 dark:to-violet-400">
              Actually Break.
            </span>
          </h1>
          
          <p className="mx-auto mt-6 max-w-2xl text-lg text-muted-foreground sm:text-xl leading-relaxed">
            RobustLens is a rigorous statistical evaluation pipeline designed to test modern vision models under realistic distribution shifts. 
            Stop guessing if one model is better than another—prove it.
          </p>
        </motion.div>

        <motion.div
          className="mt-10 flex flex-wrap justify-center gap-4"
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3, ease: "easeOut" }}
        >
          <Link
            href="/leaderboard"
            className="group relative inline-flex h-12 items-center justify-center overflow-hidden rounded-full bg-primary px-8 text-sm font-semibold text-primary-foreground shadow-lg transition-all hover:scale-105 hover:shadow-primary/25"
          >
            <span className="relative z-10 flex items-center gap-2">
              Explore the Data
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="transition-transform group-hover:translate-x-1"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
            </span>
          </Link>
          <Link
            href="/methods"
            className="inline-flex h-12 items-center justify-center rounded-full border border-input bg-background/50 backdrop-blur-sm px-8 text-sm font-semibold text-foreground shadow-sm transition-all hover:bg-accent hover:text-accent-foreground"
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
