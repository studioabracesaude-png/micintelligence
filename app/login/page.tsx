'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';

export default function LoginPage() {
  const [username, setUsername] = useState('coach');
  const [password, setPassword] = useState('coach123');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError('');

    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });

    setLoading(false);

    if (!res.ok) {
      const data = await res.json();
      setError(data.error || 'Login failed.');
      return;
    }

    router.push('/dashboard');
    router.refresh();
  }

  return (
    <div className="mx-auto mt-16 max-w-md card">
      <h1 className="text-xl font-semibold mb-4">Coach Login</h1>
      <form className="space-y-3" onSubmit={onSubmit}>
        <input className="w-full rounded border p-2" value={username} onChange={(e) => setUsername(e.target.value)} placeholder="Username" />
        <input className="w-full rounded border p-2" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Password" />
        {error && <p className="text-sm text-red-600">{error}</p>}
        <button disabled={loading} className="w-full rounded bg-slate-900 text-white p-2 disabled:opacity-60">{loading ? 'Signing in...' : 'Sign in'}</button>
      </form>
    </div>
  );
}
