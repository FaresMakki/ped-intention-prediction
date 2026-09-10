import type { ReactNode } from "react";
import { BrandMark } from "./BrandMark";

interface AppHeaderProps {
  activePage: "live" | "comparison";
  subtitle: string;
  actions: ReactNode;
}

export function AppHeader({ activePage, subtitle, actions }: AppHeaderProps) {
  return (
    <header className="topbar">
      <div className="brand">
        <BrandMark />
        <div>
          <strong>Crossing Intelligence Lab</strong>
          <span>{subtitle}</span>
        </div>
      </div>

      <nav className="page-nav" aria-label="Application pages">
        <a className={activePage === "live" ? "active" : ""} href="#/">
          Live fusion
        </a>
        <a
          className={activePage === "comparison" ? "active" : ""}
          href="#/comparison"
        >
          Compare models
        </a>
      </nav>

      <div className="topbar-actions">{actions}</div>
    </header>
  );
}
