import { NextResponse } from 'next/server';
import bcrypt from 'bcryptjs';
import db from '@/lib/db';
import { authCookie, createSessionToken } from '@/lib/auth';

export async function POST(request: Request) {
  const { username, password } = await request.json();

  if (!username || !password) {
    return NextResponse.json({ error: 'Username and password are required.' }, { status: 400 });
  }

  const coach = db.prepare('SELECT * FROM coaches WHERE username = ?').get(username) as
    | { username: string; password_hash: string }
    | undefined;

  if (!coach || !bcrypt.compareSync(password, coach.password_hash)) {
    return NextResponse.json({ error: 'Invalid credentials.' }, { status: 401 });
  }

  const token = createSessionToken(coach.username);
  return new NextResponse(JSON.stringify({ ok: true }), {
    status: 200,
    headers: {
      'Set-Cookie': authCookie(token),
      'Content-Type': 'application/json',
    },
  });
}
