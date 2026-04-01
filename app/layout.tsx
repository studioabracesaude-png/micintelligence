import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Coach Assessments',
  description: 'Internal student assessment manager',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <main className="mx-auto max-w-7xl p-4 md:p-6">{children}</main>
      </body>
    </html>
  );
}
