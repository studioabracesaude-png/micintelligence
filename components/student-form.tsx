'use client';

import { useState } from 'react';

type Props = {
  onSaved: () => void;
};

export default function StudentForm({ onSaved }: Props) {
  const [form, setForm] = useState({ name: '', age: 18, gender: 'Other', phone: '', notes: '' });

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const res = await fetch('/api/students', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(form),
    });
    if (res.ok) {
      setForm({ name: '', age: 18, gender: 'Other', phone: '', notes: '' });
      onSaved();
    }
  }

  return (
    <form onSubmit={submit} className="card grid gap-2 md:grid-cols-2">
      <input required className="rounded border p-2" placeholder="Name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
      <input required className="rounded border p-2" type="number" min={1} placeholder="Age" value={form.age} onChange={(e) => setForm({ ...form, age: Number(e.target.value) })} />
      <select className="rounded border p-2" value={form.gender} onChange={(e) => setForm({ ...form, gender: e.target.value })}>
        <option>Female</option>
        <option>Male</option>
        <option>Other</option>
      </select>
      <input className="rounded border p-2" placeholder="Phone" value={form.phone} onChange={(e) => setForm({ ...form, phone: e.target.value })} />
      <textarea className="rounded border p-2 md:col-span-2" placeholder="Notes (health history, injuries)" value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })} />
      <button className="rounded bg-emerald-600 text-white py-2 md:col-span-2">Add student</button>
    </form>
  );
}
