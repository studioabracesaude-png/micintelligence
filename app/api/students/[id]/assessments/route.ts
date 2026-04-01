import { NextResponse } from 'next/server';
import db from '@/lib/db';
import { getCurrentCoach } from '@/lib/auth';
import { computeMovementQualityScore } from '@/lib/types';

export async function GET(_: Request, { params }: { params: Promise<{ id: string }> }) {
  const coach = await getCurrentCoach();
  if (!coach) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const { id } = await params;
  const data = db
    .prepare('SELECT * FROM assessments WHERE student_id = ? ORDER BY created_at DESC')
    .all(id);
  return NextResponse.json(data);
}

export async function POST(request: Request, { params }: { params: Promise<{ id: string }> }) {
  const coach = await getCurrentCoach();
  if (!coach) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const { id } = await params;
  const body = await request.json();
  const { score, riskFlag } = computeMovementQualityScore(body);

  db.prepare(`
    INSERT INTO assessments (
      student_id, head_position, shoulders_symmetry, thoracic_curvature, hip_alignment, knee_alignment,
      feet_positioning, ankle_dorsiflexion, hip_rotation, shoulder_flexion, thoracic_rotation,
      plank_time, single_leg_balance, squat_pattern, push_pull_quality, pain_scale,
      injury_notes, trainer_observations, movement_quality_score, risk_flag
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    id,
    body.head_position,
    body.shoulders_symmetry,
    body.thoracic_curvature,
    body.hip_alignment,
    body.knee_alignment,
    body.feet_positioning,
    body.ankle_dorsiflexion,
    body.hip_rotation,
    body.shoulder_flexion,
    body.thoracic_rotation,
    body.plank_time,
    body.single_leg_balance,
    body.squat_pattern,
    body.push_pull_quality,
    body.pain_scale,
    body.injury_notes || '',
    body.trainer_observations || '',
    score,
    riskFlag
  );

  return NextResponse.json({ ok: true, score, riskFlag }, { status: 201 });
}
