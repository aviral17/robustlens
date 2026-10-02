import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { ThemeProvider } from "@/components/theme-provider";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "RobustLens — Vision Model Robustness Under Distribution Shift",
  description:
    "Rigorous statistical evaluation of pretrained image classification models under controlled distribution shifts. Bootstrap CIs, significance tests, calibration, conformal prediction, and ensemble analysis.",
  keywords: [
    "robustness",
    "computer vision",
    "distribution shift",
    "model evaluation",
    "bootstrap confidence intervals",
    "McNemar test",
    "calibration",
    "conformal prediction",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable}`}
      suppressHydrationWarning
    >
      <body className="min-h-screen bg-background font-sans antialiased">
        <ThemeProvider defaultTheme="dark" storageKey="robustlens-theme">
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
