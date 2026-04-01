import crypto from 'crypto';
import { cookies } from 'next/headers';

const COOKIE_NAME = 'coach_session';
const SECRET = process.env.AUTH_SECRET || 'dev-secret-change-me';

export function createSessionToken(username: string) {
  const payload = `${username}:${Date.now()}`;
  const sig = crypto.createHmac('sha256', SECRET).update(payload).digest('hex');
  return Buffer.from(`${payload}:${sig}`).toString('base64url');
}

export function verifySessionToken(token?: string | null): string | null {
  if (!token) return null;
  try {
    const decoded = Buffer.from(token, 'base64url').toString('utf8');
    const [username, ts, sig] = decoded.split(':');
    if (!username || !ts || !sig) return null;
    const expected = crypto.createHmac('sha256', SECRET).update(`${username}:${ts}`).digest('hex');
    return sig === expected ? username : null;
  } catch {
    return null;
  }
}

export async function getCurrentCoach() {
  const cookieStore = await cookies();
  const token = cookieStore.get(COOKIE_NAME)?.value;
  return verifySessionToken(token);
}

export function authCookie(token: string) {
  return `${COOKIE_NAME}=${token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000`;
}

export const SESSION_COOKIE = COOKIE_NAME;
