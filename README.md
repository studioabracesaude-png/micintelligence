# Fitness Coach Assessment App

Internal-use web app for coaches to manage students and track physical assessments over time.

## Stack
- Next.js (App Router) + React hooks
- SQLite via `better-sqlite3`
- Tailwind CSS

## Features
- Coach login (no public signup)
- Student CRUD with health notes
- Multi-category movement assessment with 0-3 scoring
- Automatic movement quality score calculation
- Risk flags and re-evaluation alerts (>30 days)
- Timeline history + latest vs previous comparison

## Local run
1. Install dependencies:
   ```bash
   npm install
   ```
2. Start dev server:
   ```bash
   npm run dev
   ```
3. Open `http://localhost:3000`
4. Default credentials:
   - username: `coach`
   - password: `coach123`

You can override credentials by setting env vars before first run:
- `COACH_USERNAME`
- `COACH_PASSWORD`
- `AUTH_SECRET`

## Database schema
SQLite file: `fitness.db` (auto-created).
Tables:
- `coaches`
- `students`
- `assessments`

Schema and seed logic live in `lib/db.ts`.

## Notes
- This project is optimized for fast daily use in studio settings with card-based UI and compact forms.
- PDF export/charts were left out to keep dependencies lightweight, but the assessment timeline is in place for future enhancements.
