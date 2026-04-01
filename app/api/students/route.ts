import { NextResponse } from 'next/server';
import db from '@/lib/db';
import { getCurrentCoach } from '@/lib/auth';

export async function GET() {
  const coach = await getCurrentCoach();
  if (!coach) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const students = db.prepare('SELECT * FROM students ORDER BY created_at DESC').all();
  return NextResponse.json(students);
}

export async function POST(request: Request) {
  const coach = await getCurrentCoach();
  if (!coach) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const body = await request.json();
  const { name, age, gender, phone = '', notes = '' } = body;

  if (!name || !age || !gender) {
    return NextResponse.json({ error: 'Name, age and gender are required.' }, { status: 400 });
  }

  const result = db.prepare(
    'INSERT INTO students (name, age, gender, phone, notes, updated_at) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)'
  ).run(name, age, gender, phone, notes);

  return NextResponse.json({ id: result.lastInsertRowid }, { status: 201 });
}
