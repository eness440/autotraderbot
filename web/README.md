# Proculus Web

A separate Next.js frontend for the existing Python AutoTraderBot repository. It keeps the trading runtime untouched while adding a production-oriented React + TypeScript + Tailwind 4 + shadcn-compatible interface.

## Stack

- Next.js / React
- TypeScript
- Tailwind CSS 4
- shadcn-compatible project layout
- lucide-react icons
- `tw-animate-css`

## Component paths

The shadcn alias is configured to use `@/components/ui`, whose physical path is:

```text
web/components/ui
```

This is the intended default for this frontend, so no alternate component directory is required. Keeping shared primitives under `components/ui` matters because shadcn CLI-generated components and application imports use the same alias and can be upgraded or composed without path drift.

The integrated hero lives at:

```text
web/components/ui/responsive-hero-banner.tsx
```

The Proculus-specific demo/configuration lives at:

```text
web/components/demo.tsx
```

## Run locally

```bash
cd web
npm install
npm run dev
```

Then open `http://localhost:3000`.

## Type check / production build

```bash
npm run typecheck
npm run build
npm start
```

## shadcn CLI

`components.json` is already included. To add additional UI primitives later:

```bash
npx shadcn@latest add button card dialog sheet
```

If rebuilding the frontend from scratch instead, initialize a Next.js TypeScript project, install Tailwind 4, then run `npx shadcn@latest init`. Keep the alias target as `@/components/ui`.

## Backend integration boundary

The browser should never receive OKX API secrets or LLM API keys. Expose only an authenticated API base URL to the frontend, for example:

```bash
NEXT_PUBLIC_PROCULUS_API_URL=https://api.example.com
```

Use a server-side adapter (FastAPI is a natural fit for the existing Python project) for runtime status, positions, signals, configuration changes, pause/resume and emergency controls. Authentication, authorization, audit logs and confirmation gates should live on that server boundary.

## Design mapping

The landing page carries forward the latest Proculus design language: dark fluid surface, mint/cyan intelligence accents, visible decision layers, explicit risk governance, and a command-center transition. The hero is populated from the repository's current configuration values (Hybrid mode, 0.70 trade gate, 25x risk ceiling, 15m timeframe) rather than generic demo copy.
