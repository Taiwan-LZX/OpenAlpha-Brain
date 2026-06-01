---
snapshot_id: "auth_feature_2026-05-09"
created_at: "2026-05-09T22:00:00+08:00"
version: "3.0"

# Layer 1: Execution Register (always restored)
goal: "Implement JWT-based authentication with refresh token rotation"
state: "Debugging refresh token invalidation — tokens expire silently"
next_action: "Add token expiry check in useAuth hook, test with 60s TTL"
active_files:
  - "src/hooks/useAuth.ts"
  - "src/middleware/auth.ts"
  - "src/lib/token.ts"
blocker: "Refresh token not triggering re-auth before API calls fail"

# Layer 2: Cognitive Cache (restored on demand)
constraints:
  - "No external auth service (Auth0, Clerk) — must be self-hosted"
  - "User prefers httpOnly cookies over localStorage for tokens"
  - "Must support Safari ITP (Intelligent Tracking Prevention)"
decisions:
  - "Use access+refresh token pair, not session-based auth"
  - "Access token: 15min TTL, Refresh token: 7d TTL with rotation"
  - "Store refresh token in httpOnly secure cookie, access token in memory only"
excluded_paths:
  - "localStorage for tokens — XSS vulnerability, user explicitly rejected"
  - "Single long-lived token — security risk, doesn't follow best practices"
  - "WebSocket-based token refresh — overcomplicates, standard HTTP works fine"
---

## Layer 3: Cold Archive (debug only)

### Completed Steps
1. Created JWT utility functions (sign, verify, rotate)
2. Implemented login/register endpoints with bcrypt
3. Added refresh token model in database
4. Created auth middleware for protected routes
5. Built useAuth hook with automatic token attachment

### File Changes
 src/hooks/useAuth.ts     | 87 +++---
 src/middleware/auth.ts    | 45 ++--
 src/lib/token.ts         | 120 +++++++
 src/api/auth.ts          | 67 ++++
 prisma/schema.prisma     | 12 +-

### Recent Tool Calls
1. Edit src/hooks/useAuth.ts — Added token refresh on 401 response
2. Bash npx prisma db push — Updated database schema
3. Read src/middleware/auth.ts — Checked token verification logic
4. Write src/lib/token.ts — Created token utility with rotation support
5. Bash npm test -- --grep "auth" — Ran auth tests: 8/12 passing
