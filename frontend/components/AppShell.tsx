"use client";

import { ChatCircleDots, FilePlus, Files } from "@phosphor-icons/react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import StatusBar from "@/components/StatusBar";

const NAV = [
  { href: "/", label: "Chat", icon: ChatCircleDots },
  { href: "/documents", label: "Documents", icon: Files },
];

/**
 * Nav rail + one status bar, wrapping every page.
 *
 * The rail replaces a top nav so the chat can use full height: this is a
 * workbench, not a document to scroll. Two stacked banners previously ate
 * ~90px before any content; disclaimer and health now share one line.
 */
export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="flex min-h-dvh">
      <nav
        aria-label="Main"
        className="flex w-16 shrink-0 flex-col items-center gap-1 border-r bg-sidebar py-4 lg:w-52 lg:items-stretch lg:px-3"
      >
        <div className="mb-4 px-2 lg:px-2">
          <span className="hidden font-[family-name:var(--font-heading)] text-sm font-semibold tracking-tight lg:block">
            Medical RAG
          </span>
          <span
            aria-hidden
            className="block text-center font-[family-name:var(--font-heading)] text-base font-bold text-primary lg:hidden"
          >
            M
          </span>
        </div>

        {NAV.map(({ href, label, icon: Icon }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              aria-current={active ? "page" : undefined}
              // Current location must be visually distinct, and not by colour
              // alone — the active item also carries weight and a filled icon.
              className={`flex items-center justify-center gap-3 rounded-md px-2 py-2.5 text-sm transition-colors lg:justify-start
                ${
                  active
                    ? "bg-sidebar-accent font-semibold text-sidebar-accent-foreground"
                    : "text-muted-foreground hover:bg-sidebar-accent/60 hover:text-foreground"
                }`}
            >
              <Icon size={20} weight={active ? "fill" : "regular"} aria-hidden />
              <span className="hidden lg:inline">{label}</span>
              <span className="sr-only lg:hidden">{label}</span>
            </Link>
          );
        })}

        <Link
          href="/documents"
          className="mt-auto flex items-center justify-center gap-3 rounded-md px-2 py-2.5 text-sm text-muted-foreground transition-colors hover:bg-sidebar-accent/60 hover:text-foreground lg:justify-start"
        >
          <FilePlus size={20} aria-hidden />
          <span className="hidden lg:inline">Upload a PDF</span>
          <span className="sr-only lg:hidden">Upload a PDF</span>
        </Link>
      </nav>

      <div className="flex min-w-0 flex-1 flex-col">
        <StatusBar />
        <main className="min-h-0 flex-1">{children}</main>
      </div>
    </div>
  );
}
