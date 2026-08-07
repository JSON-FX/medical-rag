import type { Metadata } from "next";
import { Figtree, Noto_Sans } from "next/font/google";

import AppShell from "@/components/AppShell";
import "./globals.css";

// next/font downloads these at BUILD time and self-hosts the files. A
// <link> to fonts.googleapis.com would make every page load reach out to
// Google, which contradicts the one promise this app makes: nothing about
// your documents leaves the machine.
const figtree = Figtree({
  subsets: ["latin"],
  variable: "--font-heading",
  display: "swap",
});

const notoSans = Noto_Sans({
  subsets: ["latin"],
  variable: "--font-body",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Medical RAG",
  description: "Ask questions grounded in medical documents you upload.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${figtree.variable} ${notoSans.variable}`}>
      <body className="min-h-dvh bg-background font-[family-name:var(--font-body)] text-foreground antialiased">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
