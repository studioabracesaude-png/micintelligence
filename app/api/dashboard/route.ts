import { NextResponse } from 'next/server';
import db from '@/lib/db';
import { getCurrentCoach } from '@/lib/auth';

export async function GET() {
  const coach = await getCurrentCoach();
  if (!coach) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const students = db.prepare(`
    SELECT s.*, 
      MAX(a.created_at) as last_assessment_date,
      MAX(a.risk_flag) as risk_flag
    FROM students s
    LEFT JOIN assessments a ON a.student_id = s.id
    GROUP BY s.id
    ORDER BY s.name ASC
  `).all() as Array<Record<string, string | number | null>>;

  return NextResponse.json(students);
}
