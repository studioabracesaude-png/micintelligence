import { NextResponse } from 'next/server';
import db from '@/lib/db';
import { getCurrentCoach } from '@/lib/auth';

export async function GET(_: Request, { params }: { params: Promise<{ id: string }> }) {
  const coach = await getCurrentCoach();
  if (!coach) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const { id } = await params;
  const student = db.prepare('SELECT * FROM students WHERE id = ?').get(id);
  if (!student) return NextResponse.json({ error: 'Student not found' }, { status: 404 });

  return NextResponse.json(student);
}

export async function PUT(request: Request, { params }: { params: Promise<{ id: string }> }) {
  const coach = await getCurrentCoach();
  if (!coach) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const { id } = await params;
  const { name, age, gender, phone = '', notes = '' } = await request.json();

  db.prepare(
    'UPDATE students SET name = ?, age = ?, gender = ?, phone = ?, notes = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?'
  ).run(name, age, gender, phone, notes, id);

  return NextResponse.json({ ok: true });
}

export async function DELETE(_: Request, { params }: { params: Promise<{ id: string }> }) {
  const coach = await getCurrentCoach();
  if (!coach) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const { id } = await params;
  db.prepare('DELETE FROM students WHERE id = ?').run(id);
  return NextResponse.json({ ok: true });
}
