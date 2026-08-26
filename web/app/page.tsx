import HeroDemo from "@/components/demo";
import {
  Activity,
  BrainCircuit,
  Gauge,
  Layers3,
  Radar,
  ShieldCheck,
  TerminalSquare,
  Zap,
} from "lucide-react";

const flow = [
  {
    index: "01",
    title: "Observe",
    copy: "Order book, liquidation, on-chain, social and macro context enter the decision surface as independent evidence streams.",
    icon: Radar,
  },
  {
    index: "02",
    title: "Reason",
    copy: "Hybrid AI and ML layers score the setup instead of relying on one model as a single oracle.",
    icon: BrainCircuit,
  },
  {
    index: "03",
    title: "Gate",
    copy: "Candidates below the configured 0.70 confidence threshold do not qualify for execution.",
    icon: Gauge,
  },
  {
    index: "04",
    title: "Govern",
    copy: "Position limits, portfolio risk, circuit breakers and the kill-switch remain authoritative over model conviction.",
    icon: ShieldCheck,
  },
  {
    index: "05",
    title: "Execute",
    copy: "Qualified orders move through watched execution, stop protection, runtime state and persistent logging.",
    icon: Zap,
  },
];

const modelWeights = [
  ["DeepSeek", "45.0%"],
  ["ChatGPT", "40.0%"],
  ["BiLSTM", "7.5%"],
  ["PPO-RL", "7.5%"],
];

export default function Home() {
  return (
    <main>
      <HeroDemo />

      <section id="system" className="mx-auto max-w-7xl px-6 py-24 sm:py-32">
        <div className="max-w-3xl">
          <p className="text-xs font-semibold tracking-[0.24em] text-[#b8ffe8]/60">01 / DECISION FLOW</p>
          <h2 className="mt-5 text-4xl tracking-[-0.035em] text-white sm:text-5xl">
            A trading bot is a script. <span className="font-serif text-[#b8ffe8]">Proculus is a decision system.</span>
          </h2>
          <p className="mt-6 max-w-2xl text-base leading-7 text-white/55">
            The interface exposes the same separation that already exists in the bot: observation, inference, calibration, risk, execution and monitoring remain visible instead of collapsing into one opaque score.
          </p>
        </div>

        <div className="mt-14 grid gap-3 md:grid-cols-5">
          {flow.map((step) => {
            const Icon = step.icon;
            return (
              <article key={step.index} className="group rounded-[1.75rem] border border-white/8 bg-white/[0.025] p-5 transition-colors hover:border-[#b8ffe8]/20 hover:bg-white/[0.04]">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs text-white/30">{step.index}</span>
                  <Icon className="h-4 w-4 text-[#b8ffe8]/65" />
                </div>
                <h3 className="mt-8 text-lg font-semibold text-white/90">{step.title}</h3>
                <p className="mt-3 text-sm leading-6 text-white/45">{step.copy}</p>
              </article>
            );
          })}
        </div>
      </section>

      <section id="intelligence" className="border-y border-white/8 bg-white/[0.018]">
        <div className="mx-auto grid max-w-7xl gap-12 px-6 py-24 lg:grid-cols-[1fr_.9fr] lg:items-center sm:py-32">
          <div>
            <p className="text-xs font-semibold tracking-[0.24em] text-[#7fe8ff]/60">02 / INTELLIGENCE</p>
            <h2 className="mt-5 text-4xl tracking-[-0.035em] sm:text-5xl">
              No single oracle. <span className="font-serif text-[#7fe8ff]">A controlled ensemble.</span>
            </h2>
            <p className="mt-6 max-w-xl text-base leading-7 text-white/55">
              The current repository configuration runs a hybrid decision mode and explicitly weights parallel reasoning and sequence/policy models. The website surfaces those weights instead of hiding them behind a generic “AI” label.
            </p>
          </div>

          <div className="rounded-[2rem] border border-white/8 bg-[#07100e] p-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-white/8 px-3 pb-4">
              <div className="flex items-center gap-2 text-sm font-semibold text-white/80"><Layers3 className="h-4 w-4 text-[#7fe8ff]" /> Hybrid weights</div>
              <span className="rounded-full border border-[#9cffbd]/20 bg-[#9cffbd]/5 px-3 py-1 font-mono text-[10px] text-[#9cffbd]">MODE / HYBRID</span>
            </div>
            <div className="mt-3 space-y-2">
              {modelWeights.map(([name, value]) => (
                <div key={name} className="grid grid-cols-[110px_1fr_60px] items-center gap-3 rounded-2xl border border-white/6 bg-white/[0.025] px-4 py-3">
                  <span className="text-sm text-white/70">{name}</span>
                  <div className="h-1.5 overflow-hidden rounded-full bg-white/8">
                    <div className="h-full rounded-full bg-[linear-gradient(90deg,#b8ffe8,#7fe8ff)]" style={{ width: value }} />
                  </div>
                  <span className="text-right font-mono text-xs text-white/45">{value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section id="risk" className="mx-auto max-w-7xl px-6 py-24 sm:py-32">
        <div className="grid gap-12 lg:grid-cols-[.9fr_1.1fr] lg:items-center">
          <div className="order-2 rounded-[2.25rem] border border-[#b8ffe8]/10 bg-[radial-gradient(circle_at_50%_35%,rgba(184,255,232,.10),transparent_40%),#07100e] p-7 lg:order-1">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold tracking-[0.2em] text-white/35">RISK GOVERNOR</span>
              <ShieldCheck className="h-5 w-5 text-[#9cffbd]" />
            </div>
            <div className="mt-10 grid grid-cols-2 gap-3">
              {[
                ["Max leverage", "25×"],
                ["Max position", "10%"],
                ["Portfolio risk", "5%"],
                ["Daily loss limit", "5%"],
                ["Kill / reduce", "-3%"],
                ["Kill / stop", "-7%"],
              ].map(([label, value]) => (
                <div key={label} className="rounded-2xl border border-white/7 bg-black/20 p-4">
                  <div className="text-xs text-white/35">{label}</div>
                  <div className="mt-2 font-mono text-xl text-white/85">{value}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="order-1 lg:order-2">
            <p className="text-xs font-semibold tracking-[0.24em] text-[#9cffbd]/60">03 / RISK</p>
            <h2 className="mt-5 text-4xl tracking-[-0.035em] sm:text-5xl">
              Model conviction does not outrank <span className="font-serif text-[#9cffbd]">capital protection.</span>
            </h2>
            <p className="mt-6 max-w-xl text-base leading-7 text-white/55">
              The site makes the hard limits legible at a glance. High confidence may change a candidate’s priority, but risk ceilings, circuit breakers and kill states remain separate controls.
            </p>
          </div>
        </div>
      </section>

      <section id="models" className="border-y border-white/8 bg-white/[0.018]">
        <div className="mx-auto max-w-7xl px-6 py-24 sm:py-32">
          <div className="grid gap-4 md:grid-cols-3">
            {[
              { icon: BrainCircuit, title: "Reasoning layer", copy: "Parallel LLM inputs contribute structured decisions without becoming execution authority." },
              { icon: Activity, title: "Model layer", copy: "Sequence and policy models remain measurable, replaceable and independently observable." },
              { icon: ShieldCheck, title: "Risk layer", copy: "Explicit limits and breakers decide whether model output can become capital exposure." },
            ].map((item) => {
              const Icon = item.icon;
              return (
                <article key={item.title} className="rounded-[2rem] border border-white/8 bg-[#07100e] p-7">
                  <Icon className="h-5 w-5 text-[#c7b8ff]" />
                  <h3 className="mt-10 text-xl font-semibold">{item.title}</h3>
                  <p className="mt-3 text-sm leading-6 text-white/45">{item.copy}</p>
                </article>
              );
            })}
          </div>
        </div>
      </section>

      <section id="command-center" className="mx-auto max-w-7xl px-6 py-24 sm:py-32">
        <div className="overflow-hidden rounded-[2.5rem] border border-white/10 bg-[#07100e] shadow-[0_40px_120px_rgba(0,0,0,.4)]">
          <div className="flex flex-col gap-4 border-b border-white/8 px-6 py-5 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-3">
              <TerminalSquare className="h-5 w-5 text-[#b8ffe8]" />
              <div>
                <div className="text-sm font-semibold tracking-wide">PROCULUS / COMMAND CENTER</div>
                <div className="mt-0.5 text-xs text-white/35">Frontend integration surface</div>
              </div>
            </div>
            <div className="flex items-center gap-2 text-[11px]">
              <span className="rounded-full border border-[#9cffbd]/15 bg-[#9cffbd]/5 px-3 py-1.5 font-mono text-[#9cffbd]">● ENGINE / READY</span>
              <span className="rounded-full border border-white/8 px-3 py-1.5 font-mono text-white/45">OKX / DEMO</span>
            </div>
          </div>

          <div className="grid gap-px bg-white/8 lg:grid-cols-[1.2fr_.8fr]">
            <div className="bg-[#07100e] p-6 sm:p-8">
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                {[
                  ["Trade gate", "0.70"],
                  ["Timeframe", "15m"],
                  ["Max positions", "5"],
                  ["AI mode", "Hybrid"],
                ].map(([label, value]) => (
                  <div key={label} className="rounded-2xl border border-white/7 bg-white/[0.025] p-4">
                    <div className="text-[10px] uppercase tracking-[0.15em] text-white/30">{label}</div>
                    <div className="mt-3 font-mono text-lg text-white/85">{value}</div>
                  </div>
                ))}
              </div>

              <div className="mt-4 rounded-3xl border border-white/7 bg-black/20 p-5">
                <div className="mb-5 flex items-center justify-between">
                  <span className="text-xs font-semibold tracking-[0.16em] text-white/40">DECISION STREAM</span>
                  <Activity className="h-4 w-4 text-[#b8ffe8]" />
                </div>
                <div className="space-y-3 font-mono text-xs">
                  <div className="grid grid-cols-[70px_90px_1fr] gap-3 text-white/40"><span>09:42:12</span><span className="text-[#7fe8ff]">OBSERVE</span><span>market context synchronized</span></div>
                  <div className="grid grid-cols-[70px_90px_1fr] gap-3 text-white/40"><span>09:42:13</span><span className="text-[#c7b8ff]">INFER</span><span>hybrid layer produced candidate score</span></div>
                  <div className="grid grid-cols-[70px_90px_1fr] gap-3 text-white/40"><span>09:42:13</span><span className="text-[#9cffbd]">RISK</span><span>limits checked before execution handoff</span></div>
                </div>
              </div>
            </div>

            <div className="bg-[#060b0a] p-6 sm:p-8">
              <p className="text-xs font-semibold tracking-[0.2em] text-white/35">BACKEND BOUNDARY</p>
              <h3 className="mt-5 text-2xl tracking-[-0.02em]">Browser controls the interface. <span className="font-serif text-[#b8ffe8]">Secrets stay server-side.</span></h3>
              <p className="mt-4 text-sm leading-6 text-white/45">
                This frontend is intentionally separated from exchange and model credentials. Connect it to an authenticated API adapter on the PC/server running the bot; never expose OKX or LLM secrets through NEXT_PUBLIC variables.
              </p>
              <div className="mt-7 flex items-center gap-2 rounded-2xl border border-white/7 bg-white/[0.025] px-4 py-3 font-mono text-xs text-white/45">
                <span className="text-[#9cffbd]">API</span>
                <span>NEXT_PUBLIC_PROCULUS_API_URL</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <footer className="mx-auto flex max-w-7xl flex-col gap-3 border-t border-white/8 px-6 py-8 text-xs text-white/30 sm:flex-row sm:items-center sm:justify-between">
        <span>PROCULUS / Adaptive Futures Intelligence</span>
        <span>Observable automation. Human authority.</span>
        <span>Interface layer — not financial advice.</span>
      </footer>
    </main>
  );
}
