import { useEffect, useState } from "react";
import { ComparisonPage } from "./pages/ComparisonPage";
import { LiveFusionPage } from "./pages/LiveFusionPage";

type Page = "live" | "comparison";

function pageFromHash(): Page {
  return window.location.hash === "#/comparison" ? "comparison" : "live";
}

function App() {
  const [page, setPage] = useState<Page>(pageFromHash);

  useEffect(() => {
    const updatePage = () => setPage(pageFromHash());
    window.addEventListener("hashchange", updatePage);
    return () => window.removeEventListener("hashchange", updatePage);
  }, []);

  return page === "comparison" ? <ComparisonPage /> : <LiveFusionPage />;
}

export default App;
