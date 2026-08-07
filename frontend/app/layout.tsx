import type { Metadata } from "next";
import Link from "next/link";
import DisclaimerBanner from "@/components/DisclaimerBanner";
import "./globals.css";

export const metadata: Metadata = {
  title: "Medical RAG",
  description: "Ask questions grounded in medical documents you upload.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background text-foreground antialiased">
        <DisclaimerBanner />
        <nav className="flex gap-4 border-b px-4 py-3 text-sm">
          <Link href="/" className="font-medium hover:underline">Chat</Link>
          <Link href="/documents" className="font-medium hover:underline">Documents</Link>
        </nav>
        <main className="mx-auto w-full max-w-3xl px-4 py-6">{children}</main>
      </body>
    </html>
  );
}
