'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import StudentForm from '@/components/student-form';

type Student = {
  id: number;
  name: string;
  age: number;
  gender: string;
  last_assessment_date: string | null;
  risk_flag: number;
};

function daysSince(date: string | null) {
  if (!date) return 999;
  return Math.floor((Date.now() - new Date(date).getTime()) / (1000 * 60 * 60 * 24));
}

export default function DashboardPage() {
  const [students, setStudents] = useState<Student[]>([]);

  async function load() {
    const res = await fetch('/api/dashboard');
    if (res.ok) setStudents(await res.json());
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Student Dashboard</h1>
        <p className="text-sm text-slate-500">{students.length} active students</p>
      </div>

      <StudentForm onSaved={load} />

      <div className="grid gap-3">
        {students.map((s) => {
          const stale = daysSince(s.last_assessment_date) > 30;
          return (
            <Link key={s.id} href={`/students/${s.id}`} className="card flex items-center justify-between hover:border-slate-300">
              <div>
                <h2 className="font-medium">{s.name}</h2>
                <p className="text-sm text-slate-600">{s.age} y/o • {s.gender}</p>
                <p className="text-xs text-slate-500">Last assessment: {s.last_assessment_date ? new Date(s.last_assessment_date).toLocaleDateString() : 'No assessments yet'}</p>
              </div>
              <div className="text-right">
                {s.risk_flag ? <p className="text-red-600 text-sm font-medium">Risk flag</p> : <p className="text-emerald-600 text-sm">Stable</p>}
                {stale && <p className="text-amber-600 text-xs">Re-evaluation due</p>}
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
