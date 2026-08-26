import React from "react";
import ResponsiveHeroBanner from "@/components/ui/responsive-hero-banner";

const HeroDemo = () => {
  return (
    <ResponsiveHeroBanner
      badgeLabel="LIVE ARCHITECTURE"
      badgeText="OKX Futures · Hybrid decision stack"
      title="Trade the system."
      titleLine2="Not the noise."
      description="Proculus combines market structure, AI reasoning, machine learning and external context, then forces every candidate through explicit risk gates before capital can move."
      primaryButtonText="Open command center"
      primaryButtonHref="#command-center"
      secondaryButtonText="Follow decision flow"
      secondaryButtonHref="#system"
      ctaButtonText="Launch console"
      ctaButtonHref="#command-center"
      navLinks={[
        { label: "System", href: "#system", isActive: true },
        { label: "Intelligence", href: "#intelligence" },
        { label: "Risk", href: "#risk" },
        { label: "Models", href: "#models" },
      ]}
      telemetry={[
        { label: "DECISION MODE", value: "HYBRID" },
        { label: "TRADE GATE", value: "≥ 0.70" },
        { label: "RISK CEILING", value: "25×" },
        { label: "TIMEFRAME", value: "15m" },
      ]}
      partnersTitle="Core decision layers"
      partners={[
        { label: "MARKET DATA", href: "#system" },
        { label: "AI REASONING", href: "#intelligence" },
        { label: "ML ENSEMBLE", href: "#models" },
        { label: "RISK GOVERNOR", href: "#risk" },
        { label: "OKX EXECUTION", href: "#command-center" },
      ]}
    />
  );
};

export default HeroDemo;
