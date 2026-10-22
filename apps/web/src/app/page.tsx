"use client";

import { motion } from "motion/react";
import Link from "next/link";
import { ArrowRight, Activity, Zap, BarChart, Shuffle, Scale, ShieldCheck, Layers } from "lucide-react";
import TryDemo from "@/components/TryDemo";

const features = [
  {
    title: "Compound Order Effects",
    description: "Evaluating all 3! = 6 permutations under nominal and matched-fidelity regimes across 973,400 forward passes.",
    icon: <Shuffle className="h-6 w-6 text-blue-500" />,
  },
  {
    title: "5-State Prediction Churn",
    description: "Decomposing model volatility into correctness churn (16.93%) and hidden error migration (23.85%).",
    icon: <Activity className="h-6 w-6 text-violet-500" />,
  },
  {
    title: "Saturated Decomposition",
    description: "Full-rank orthogonal linear parameterization (det(X) = -6) isolating operator positions and parity contrasts.",
    icon: <Layers className="h-6 w-6 text-emerald-500" />,
  },
  {
    title: "Clustered Bootstrap CIs",
    description: "B = 5,000 image-level clustered resampling iterations on N = 3,925 physical image clusters.",
    icon: <BarChart className="h-6 w-6 text-cyan-500" />,
  },
  {
    title: "Uncertainty Calibration",
    description: "10-bin Expected Calibration Error (ECE), reliability diagrams, and post-hoc temperature scaling under shift.",
    icon: <Zap className="h-6 w-6 text-amber-500" />,
  },
  {
    title: "Conformal Sets & Guarantees",
    description: "Distribution-free split conformal prediction and empirical testing of the exchangeability boundary.",
    icon: <ShieldCheck className="h-6 w-6 text-rose-500" />,
  },
];

export default function HomePage() {
  return (
    <main className="flex min-h-screen flex-col items-center pb-20 pt-32">
      {/* Decorative Background Glow */}
      <div className="absolute inset-0 -z-10 h-full w-full bg-background bg-[radial-gradient(#e5e7eb_1px,transparent_1px)] [background-size:16px_16px] dark:bg-[radial-gradient(#1f2937_1px,transparent_1px)] [mask-image:radial-gradient(ellipse_50%_50%_at_50%_50%,#000_70%,transparent_100%)]"></div>
      
      {/* Hero */}
      <section className="relative flex w-full flex-col items-center justify-center px-6 pt-10 pb-16 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: "easeOut" }}
          className="max-w-4xl"
        >
          <div className="mb-6 flex items-center justify-center gap-2">
            <span className="rounded-full bg-primary/10 px-4 py-1.5 text-sm font-bold text-primary ring-1 ring-primary/20">
              Research & Evaluation Framework
            </span>
          </div>
          
          <h1 className="text-5xl font-extrabold tracking-tight sm:text-6xl md:text-7xl lg:text-8xl font-[family-name:var(--font-poppins)]">
            <span className="bg-gradient-to-br from-foreground via-foreground/80 to-muted-foreground bg-clip-text text-transparent">
              See How Models
            </span>
            <br />
            <span className="bg-gradient-to-r from-blue-600 to-violet-600 bg-clip-text text-transparent dark:from-blue-400 dark:to-violet-400">
              Actually Break.
            </span>
          </h1>
          
          <p className="mx-auto mt-6 max-w-[800px] text-xl text-muted-foreground leading-relaxed font-medium">
            RobustLens is a rigorous statistical evaluation pipeline designed to test modern vision models under realistic distribution shifts. 
            Stop guessing if one model is better than another—prove it.
          </p>

          {/* Quick Metric Badges */}
          <div className="mt-8 grid grid-cols-2 sm:grid-cols-4 gap-3 max-w-3xl mx-auto w-full text-left">
            <div className="p-3.5 rounded-2xl bg-card/80 border border-border/50 flex flex-col gap-0.5 backdrop-blur-sm">
              <span className="text-xl sm:text-2xl font-extrabold text-blue-500">973,400</span>
              <span className="text-[11px] sm:text-xs text-muted-foreground font-medium">Forward Passes</span>
            </div>
            <div className="p-3.5 rounded-2xl bg-card/80 border border-border/50 flex flex-col gap-0.5 backdrop-blur-sm">
              <span className="text-xl sm:text-2xl font-extrabold text-violet-500">3,925</span>
              <span className="text-[11px] sm:text-xs text-muted-foreground font-medium">Validation Images</span>
            </div>
            <div className="p-3.5 rounded-2xl bg-card/80 border border-border/50 flex flex-col gap-0.5 backdrop-blur-sm">
              <span className="text-xl sm:text-2xl font-extrabold text-emerald-500">63.18%</span>
              <span className="text-[11px] sm:text-xs text-muted-foreground font-medium">Range Contraction</span>
            </div>
            <div className="p-3.5 rounded-2xl bg-card/80 border border-border/50 flex flex-col gap-0.5 backdrop-blur-sm">
              <span className="text-xl sm:text-2xl font-extrabold text-amber-500">40.78%</span>
              <span className="text-[11px] sm:text-xs text-muted-foreground font-medium">Label Churn</span>
            </div>
          </div>
        </motion.div>

        <motion.div
          className="mt-8 flex flex-wrap justify-center gap-4"
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3, ease: "easeOut" }}
        >
          <Link
            href="/order-effects"
            className="group relative inline-flex h-14 items-center justify-center overflow-hidden rounded-full bg-primary px-8 text-base font-bold text-primary-foreground shadow-lg transition-all hover:scale-105 hover:shadow-primary/25"
          >
            <span className="relative z-10 flex items-center gap-2">
              Explore Order Effects
              <ArrowRight className="h-5 w-5 transition-transform group-hover:translate-x-1" />
            </span>
          </Link>
          <Link
            href="/explorer"
            className="inline-flex h-14 items-center justify-center rounded-full border border-input bg-background/50 backdrop-blur-sm px-8 text-base font-semibold text-foreground shadow-sm transition-all hover:bg-accent hover:text-accent-foreground"
          >
            Interactive Demo
          </Link>
          <Link
            href="/methods"
            className="inline-flex h-14 items-center justify-center rounded-full border border-input bg-background/50 backdrop-blur-sm px-8 text-base font-semibold text-foreground shadow-sm transition-all hover:bg-accent hover:text-accent-foreground"
          >
            Methodology
          </Link>
        </motion.div>
      </section>

      {/* Visual Preview Section */}
      <section className="mx-auto w-full max-w-[1400px] px-6 py-16">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="rounded-[2.5rem] bg-card border border-border/50 p-8 shadow-xl md:p-12"
        >
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold tracking-tight md:text-4xl">The Anatomy of a Distribution Shift</h2>
            <p className="mt-4 text-lg text-muted-foreground">How image corruptions impact model prediction certainty.</p>
          </div>
          
          <div className="flex flex-col md:flex-row items-center justify-between gap-8 md:gap-4 relative">
            
            {/* Clean Image */}
            <div className="flex flex-col items-center gap-4 w-full md:w-1/3">
              <div className="w-full max-w-[280px] aspect-square rounded-2xl bg-muted overflow-hidden relative border shadow-sm flex items-center justify-center">
                <img src="/examples/img_4_clean.jpg" alt="Clean Hammerhead" className="absolute inset-0 w-full h-full object-cover" />
                <span className="text-sm font-medium text-muted-foreground z-10 relative bg-background/80 px-3 py-1 rounded-full top-[-100px]">Clean Image</span>
              </div>
              <div className="text-center">
                <p className="font-semibold text-lg text-foreground">ResNet-50</p>
                <div className="inline-flex items-center gap-2 mt-2 bg-green-500/10 text-green-600 dark:text-green-400 px-3 py-1 rounded-full text-sm font-bold">
                  <span>Hammerhead</span>
                  <span>97% Conf</span>
                </div>
              </div>
            </div>

            {/* Shift Indicator */}
            <div className="flex flex-col items-center gap-2 text-muted-foreground">
              <div className="hidden md:flex h-0.5 w-16 bg-gradient-to-r from-transparent via-border to-transparent"></div>
              <div className="rounded-full bg-accent p-3 border shadow-sm">
                <Zap className="h-6 w-6 text-amber-500" />
              </div>
              <p className="font-semibold text-sm uppercase tracking-wider">Gaussian Noise</p>
              <p className="text-xs font-medium">Severity 3</p>
              <div className="hidden md:flex h-0.5 w-16 bg-gradient-to-r from-transparent via-border to-transparent"></div>
            </div>

            {/* Corrupted Image */}
            <div className="flex flex-col items-center gap-4 w-full md:w-1/3">
              <div className="w-full max-w-[280px] aspect-square rounded-2xl bg-muted overflow-hidden relative border shadow-sm flex items-center justify-center">
                <img src="/examples/img_4_gaussian_noise_s3.jpg" alt="Corrupted Hammerhead" className="absolute inset-0 w-full h-full object-cover" />
                <span className="text-sm font-medium text-muted-foreground z-10 relative bg-background/80 px-3 py-1 rounded-full top-[-100px]">Corrupted</span>
              </div>
              <div className="text-center">
                <p className="font-semibold text-lg text-foreground">ResNet-50</p>
                <div className="inline-flex items-center gap-2 mt-2 bg-red-500/10 text-red-600 dark:text-red-400 px-3 py-1 rounded-full text-sm font-bold">
                  <span>Swimming Trunks</span>
                  <span>76% Conf</span>
                </div>
              </div>
            </div>

          </div>
        </motion.div>
      </section>

      <TryDemo />

      {/* Features Grid */}
      <section className="mx-auto w-full max-w-[1400px] px-6 py-16">
        <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((feature, i) => (
            <motion.div
              key={feature.title}
              className="group rounded-3xl border border-border/50 bg-card p-8 transition-all hover:bg-accent/40 hover:shadow-lg"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                duration: 0.4,
                delay: 0.5 + i * 0.1,
                ease: "easeOut",
              }}
            >
              <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-primary/10">
                {feature.icon}
              </div>
              <h3 className="text-2xl font-bold">{feature.title}</h3>
              <p className="mt-3 text-base text-muted-foreground leading-relaxed">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="mt-auto border-t w-full py-8 text-center text-sm text-muted-foreground">
        <p>RobustLens — Built with PyTorch, timm, Next.js, and rigorous statistics.</p>
        <p className="mt-2">
          <Link href="/about" className="font-medium hover:text-foreground hover:underline">
            About this project
          </Link>
        </p>
      </footer>
    </main>
  );
}
