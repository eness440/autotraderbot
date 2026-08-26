import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Proculus — Adaptive Futures Intelligence",
  description:
    "Observable crypto futures decision infrastructure with layered intelligence, risk governance and operator control.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
