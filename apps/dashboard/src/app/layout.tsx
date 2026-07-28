import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import Link from 'next/link';
import { Activity, Database, LayoutDashboard } from 'lucide-react';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Continuum Dashboard',
  description: 'Mission control for your embedding models',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-background text-foreground min-h-screen flex`}>
        {/* Sidebar */}
        <aside className="w-64 border-r border-border bg-card flex flex-col">
          <div className="p-6">
            <h1 className="text-xl font-bold tracking-tighter flex items-center gap-2">
              <div className="size-4 rounded-full bg-primary animate-pulse" />
              Continuum
            </h1>
          </div>
          <nav className="flex-1 px-4 space-y-2">
            <Link
              href="/"
              className="flex items-center gap-3 px-3 py-2 rounded-md hover:bg-accent text-sm font-medium transition-colors"
            >
              <LayoutDashboard className="size-4" />
              Overview
            </Link>
            <Link
              href="/models"
              className="flex items-center gap-3 px-3 py-2 rounded-md hover:bg-accent text-sm font-medium transition-colors"
            >
              <Database className="size-4" />
              Model Registry
            </Link>
            <Link
              href="/training"
              className="flex items-center gap-3 px-3 py-2 rounded-md hover:bg-accent text-sm font-medium transition-colors"
            >
              <Activity className="size-4" />
              Training Monitor
            </Link>
          </nav>
        </aside>

        {/* Main Content */}
        <main className="flex-1 flex flex-col overflow-hidden">
          <header className="h-16 border-b border-border flex items-center px-8 bg-card/50 backdrop-blur-sm z-10">
            <h2 className="text-sm font-medium text-muted-foreground">Continuum / Dashboard</h2>
          </header>
          <div className="flex-1 overflow-y-auto p-8">{children}</div>
        </main>
      </body>
    </html>
  );
}
