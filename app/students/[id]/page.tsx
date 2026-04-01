'use client';

import { useEffect, useMemo, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';

const scoreFields = [
  ['head_position', 'Head position'],
  ['shoulders_symmetry', 'Shoulders symmetry'],
  ['thoracic_curvature', 'Thoracic curvature'],
  ['hip_alignment', 'Hip alignment'],
  ['knee_alignment', 'Knee alignment'],
  ['feet_positioning', 'Feet positioning'],
  ['ankle_dorsiflexion', 'Ankle dorsiflexion'],
  ['hip_rotation', 'Hip internal/external rotation'],
  ['shoulder_flexion', 'Shoulder flexion'],
  ['thoracic_rotation', 'Thoracic rotation'],
  ['plank_time', 'Plank quality/time'],
  ['single_leg_balance', 'Single-leg balance'],
  ['squat_pattern', 'Squat pattern'],
  ['push_pull_quality', 'Push/Pull quality'],
] as const;

const initialForm: Record<string, number | string> = Object.fromEntries(scoreFields.map(([k]) => [k, 2]));
initialForm.pain_scale = 0;
initialForm.injury_notes = '';
initialForm.trainer_observations = '';

export default function StudentPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const [student, setStudent] = useState<any>(null);
  const [assessments, setAssessments] = useState<any[]>([]);
  const [form, setForm] = useState(initialForm);

  async function load() {
    const [studentRes, aRes] = await Promise.all([
      fetch(`/api/students/${params.id}`),
      fetch(`/api/students/${params.id}/assessments`),
    ]);
    if (studentRes.ok) setStudent(await studentRes.json());
    if (aRes.ok) setAssessments(await aRes.json());
  }

  useEffect(() => {
    load();
  }, [params.id]);

  const latest = assessments[0];
  const previous = assessments[1];

  const trend = useMemo(() => {
    if (!latest || !previous) return null;
    const diff = Number(latest.movement_quality_score) - Number(previous.movement_quality_score);
    return diff;
  }, [latest, previous]);

  async function submitAssessment(e: React.FormEvent) {
    e.preventDefault();
    const res = await fetch(`/api/students/${params.id}/assessments`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(form),
    });
    if (res.ok) {
      setForm(initialForm);
      await load();
    }
  }

  async function deleteStudent() {
    if (!confirm('Delete this student and all assessments?')) return;
    const res = await fetch(`/api/students/${params.id}`, { method: 'DELETE' });
    if (res.ok) router.push('/dashboard');
  }

  if (!student) return <p>Loading...</p>;

  return (
    <div className="space-y-4 pb-8">
      <div className="card flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-semibold">{student.name}</h1>
          <p className="text-sm text-slate-600">{student.age} y/o • {student.gender} • {student.phone || 'No phone'}</p>
          <p className="text-sm text-slate-600 mt-1">{student.notes || 'No notes'}</p>
        </div>
        <button className="text-red-600 text-sm" onClick={deleteStudent}>Delete student</button>
      </div>

      {latest && (
        <div className="card">
          <h2 className="font-semibold">Progress snapshot</h2>
          <p className="text-sm">Latest movement quality score: <b>{latest.movement_quality_score}</b></p>
          <p className="text-sm">Pain scale: <b>{latest.pain_scale}/10</b></p>
          {trend !== null && <p className={`text-sm ${trend >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>Change vs previous: {trend >= 0 ? '+' : ''}{trend.toFixed(1)}</p>}
        </div>
      )}

      <form onSubmit={submitAssessment} className="card grid gap-3 md:grid-cols-2">
        <h2 className="md:col-span-2 font-semibold">New assessment</h2>
        {scoreFields.map(([key, label]) => (
          <label key={key} className="text-sm">
            {label}
            <select className="w-full border rounded p-2 mt-1" value={String(form[key])} onChange={(e) => setForm({ ...form, [key]: Number(e.target.value) })}>
              <option value="0">0 - Poor</option>
              <option value="1">1 - Limited</option>
              <option value="2">2 - OK</option>
              <option value="3">3 - Good</option>
            </select>
          </label>
        ))}

        <label className="text-sm">
          Pain scale (0-10)
          <input type="number" min={0} max={10} className="w-full border rounded p-2 mt-1" value={String(form.pain_scale)} onChange={(e) => setForm({ ...form, pain_scale: Number(e.target.value) })} />
        </label>

        <label className="text-sm md:col-span-2">
          Injury notes
          <textarea className="w-full border rounded p-2 mt-1" value={String(form.injury_notes)} onChange={(e) => setForm({ ...form, injury_notes: e.target.value })} />
        </label>

        <label className="text-sm md:col-span-2">
          Trainer observations
          <textarea className="w-full border rounded p-2 mt-1" value={String(form.trainer_observations)} onChange={(e) => setForm({ ...form, trainer_observations: e.target.value })} />
        </label>

        <button className="md:col-span-2 rounded bg-slate-900 text-white py-2">Save assessment</button>
      </form>

      <div className="card">
        <h2 className="font-semibold mb-2">Assessment timeline</h2>
        <div className="space-y-2">
          {assessments.map((a) => (
            <div key={a.id} className="rounded border p-3 text-sm">
              <p><b>{new Date(a.created_at).toLocaleString()}</b></p>
              <p>Movement quality score: {a.movement_quality_score}</p>
              <p>Pain: {a.pain_scale}/10 {a.risk_flag ? '• ⚠️ Risk flag' : ''}</p>
              <p className="text-slate-600">{a.trainer_observations || 'No observation'}</p>
            </div>
          ))}
          {assessments.length === 0 && <p className="text-sm text-slate-500">No assessments yet.</p>}
        </div>
      </div>
    </div>
  );
}
