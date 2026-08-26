"use client";

import React, { useState } from "react";
import { ArrowRight, ArrowUpRight, Menu, Play, X } from "lucide-react";

interface NavLink {
  label: string;
  href: string;
  isActive?: boolean;
}

interface Partner {
  logoUrl?: string;
  label?: string;
  href: string;
}

interface ResponsiveHeroBannerProps {
  logoUrl?: string;
  logoText?: string;
  backgroundImageUrl?: string;
  navLinks?: NavLink[];
  ctaButtonText?: string;
  ctaButtonHref?: string;
  badgeText?: string;
  badgeLabel?: string;
  title?: string;
  titleLine2?: string;
  description?: string;
  primaryButtonText?: string;
  primaryButtonHref?: string;
  secondaryButtonText?: string;
  secondaryButtonHref?: string;
  partnersTitle?: string;
  partners?: Partner[];
  telemetry?: Array<{ label: string; value: string }>;
}

const ResponsiveHeroBanner: React.FC<ResponsiveHeroBannerProps> = ({
  logoUrl,
  logoText = "PROCULUS",
  backgroundImageUrl =
    "https://images.unsplash.com/photo-1639322537228-f710d846310a?auto=format&fit=crop&w=2400&q=85",
  navLinks = [
    { label: "System", href: "#system", isActive: true },
    { label: "Intelligence", href: "#intelligence" },
    { label: "Risk", href: "#risk" },
    { label: "Models", href: "#models" },
  ],
  ctaButtonText = "Open command center",
  ctaButtonHref = "#command-center",
  badgeLabel = "LIVE",
  badgeText = "Adaptive futures intelligence",
  title = "Trade the system.",
  titleLine2 = "Not the noise.",
  description =
    "A layered futures decision stack that combines market structure, AI reasoning, machine learning and explicit risk gates before capital can move.",
  primaryButtonText = "Enter command center",
  primaryButtonHref = "#command-center",
  secondaryButtonText = "Follow decision flow",
  secondaryButtonHref = "#system",
  partnersTitle = "Core decision layers",
  partners = [],
  telemetry = [],
}) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <section className="isolate relative min-h-screen w-full overflow-hidden bg-[#050807]">
      <img
        src={backgroundImageUrl}
        alt=""
        className="absolute inset-0 h-full w-full object-cover opacity-45 saturate-50"
      />
      <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(5,8,7,.25),rgba(5,8,7,.72)_56%,#050807)]" />
      <div className="proculus-grid absolute inset-0 opacity-60 [mask-image:linear-gradient(to_bottom,black,transparent_82%)]" />
      <div className="liquid-halo pointer-events-none absolute left-1/2 top-[34%] h-80 w-80 -translate-x-1/2 rounded-full bg-[radial-gradient(circle,rgba(184,255,232,.28),rgba(127,232,255,.08)_42%,transparent_72%)]" />
      <div className="pointer-events-none absolute inset-0 ring-1 ring-white/5" />

      <header className="relative z-20">
        <div className="mx-auto max-w-7xl px-6 pt-5">
          <div className="flex items-center justify-between">
            <a href="#" className="inline-flex h-11 items-center gap-3 rounded-full border border-white/10 bg-black/20 px-4 backdrop-blur-xl">
              {logoUrl ? (
                <span
                  className="h-6 w-24 bg-contain bg-center bg-no-repeat"
                  style={{ backgroundImage: `url(${logoUrl})` }}
                />
              ) : (
                <>
                  <span className="relative h-5 w-5 rounded-full border border-[#b8ffe8]/55">
                    <span className="absolute inset-[5px] rounded-full bg-[#b8ffe8] shadow-[0_0_18px_rgba(184,255,232,.8)]" />
                  </span>
                  <span className="text-xs font-semibold tracking-[0.24em] text-white/95">{logoText}</span>
                </>
              )}
            </a>

            <nav className="hidden items-center md:flex">
              <div className="flex items-center gap-1 rounded-full border border-white/10 bg-white/[0.045] p-1 backdrop-blur-xl">
                {navLinks.map((link) => (
                  <a
                    key={link.label}
                    href={link.href}
                    className={`rounded-full px-3 py-2 text-sm font-medium transition-colors hover:bg-white/[0.06] hover:text-white ${
                      link.isActive ? "text-white" : "text-white/65"
                    }`}
                  >
                    {link.label}
                  </a>
                ))}
                <a
                  href={ctaButtonHref}
                  className="ml-1 inline-flex items-center gap-2 rounded-full bg-[#edf8f4] px-4 py-2 text-sm font-semibold text-[#07100d] transition-transform hover:scale-[1.02]"
                >
                  {ctaButtonText}
                  <ArrowUpRight className="h-4 w-4" />
                </a>
              </div>
            </nav>

            <button
              onClick={() => setMobileMenuOpen((open) => !open)}
              className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-white/10 bg-white/[0.06] backdrop-blur md:hidden"
              aria-expanded={mobileMenuOpen}
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>
          </div>

          {mobileMenuOpen && (
            <div className="mt-3 rounded-3xl border border-white/10 bg-[#08100e]/90 p-3 shadow-2xl backdrop-blur-xl md:hidden">
              {navLinks.map((link) => (
                <a
                  key={link.label}
                  href={link.href}
                  onClick={() => setMobileMenuOpen(false)}
                  className="block rounded-2xl px-4 py-3 text-sm text-white/75 hover:bg-white/[0.06] hover:text-white"
                >
                  {link.label}
                </a>
              ))}
              <a
                href={ctaButtonHref}
                onClick={() => setMobileMenuOpen(false)}
                className="mt-2 flex items-center justify-between rounded-2xl bg-[#edf8f4] px-4 py-3 text-sm font-semibold text-[#07100d]"
              >
                {ctaButtonText}
                <ArrowUpRight className="h-4 w-4" />
              </a>
            </div>
          )}
        </div>
      </header>

      <div className="relative z-10">
        <div className="mx-auto max-w-7xl px-6 pb-16 pt-28 sm:pt-32 lg:pt-40">
          <div className="mx-auto max-w-4xl text-center">
            <div className="animate-fade-slide-in-1 mb-6 inline-flex items-center gap-3 rounded-full border border-white/10 bg-white/[0.055] px-2.5 py-2 backdrop-blur-xl">
              <span className="rounded-full bg-[#dff9ef] px-2 py-0.5 text-[11px] font-bold tracking-wide text-[#082017]">
                {badgeLabel}
              </span>
              <span className="pr-1 text-sm font-medium text-white/80">{badgeText}</span>
            </div>

            <h1 className="animate-fade-slide-in-2 text-5xl font-normal leading-[0.96] tracking-[-0.04em] text-white sm:text-6xl md:text-7xl lg:text-[6.5rem]">
              <span className="font-serif">{title}</span>
              <br />
              <span className="font-serif text-[#b8ffe8]">{titleLine2}</span>
            </h1>

            <p className="animate-fade-slide-in-3 mx-auto mt-7 max-w-2xl text-base leading-7 text-white/67 sm:text-lg">
              {description}
            </p>

            <div className="animate-fade-slide-in-4 mt-10 flex flex-col items-center justify-center gap-3 sm:flex-row sm:gap-4">
              <a
                href={primaryButtonHref}
                className="inline-flex items-center gap-2 rounded-full border border-white/12 bg-white/[0.07] px-5 py-3 text-sm font-semibold text-white backdrop-blur transition-colors hover:bg-white/[0.12]"
              >
                {primaryButtonText}
                <ArrowRight className="h-4 w-4" />
              </a>
              <a
                href={secondaryButtonHref}
                className="inline-flex items-center gap-2 rounded-full px-5 py-3 text-sm font-medium text-white/75 transition-colors hover:text-white"
              >
                {secondaryButtonText}
                <Play className="h-4 w-4" />
              </a>
            </div>
          </div>

          {telemetry.length > 0 && (
            <div className="mx-auto mt-16 grid max-w-4xl grid-cols-2 overflow-hidden rounded-3xl border border-white/10 bg-black/20 backdrop-blur-xl md:grid-cols-4">
              {telemetry.map((item) => (
                <div key={item.label} className="border-white/10 px-5 py-5 text-center md:border-r last:border-r-0">
                  <div className="text-[10px] font-semibold tracking-[0.18em] text-white/40">{item.label}</div>
                  <div className="mt-2 font-mono text-lg text-white/90">{item.value}</div>
                </div>
              ))}
            </div>
          )}

          {partners.length > 0 && (
            <div className="mx-auto mt-16 max-w-5xl">
              <p className="text-center text-xs font-medium uppercase tracking-[0.2em] text-white/35">{partnersTitle}</p>
              <div className="mt-5 grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-5">
                {partners.map((partner) => (
                  <a
                    key={partner.label ?? partner.logoUrl}
                    href={partner.href}
                    className="flex min-h-14 items-center justify-center rounded-2xl border border-white/8 bg-white/[0.025] px-3 text-center text-xs font-semibold tracking-wide text-white/55 backdrop-blur transition-colors hover:border-white/15 hover:bg-white/[0.05] hover:text-white/85"
                  >
                    {partner.logoUrl ? (
                      <span className="h-8 w-full bg-contain bg-center bg-no-repeat" style={{ backgroundImage: `url(${partner.logoUrl})` }} />
                    ) : (
                      partner.label
                    )}
                  </a>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};

export default ResponsiveHeroBanner;
