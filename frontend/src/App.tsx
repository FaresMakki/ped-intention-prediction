import { useEffect, useState } from "react";
import { ComparisonPage } from "./pages/ComparisonPage";
import { LiveFusionPage } from "./pages/LiveFusionPage";
import { PreparedVideoPage } from "./pages/PreparedVideoPage";

type Page = "live" | "comparison" | "prepared-video";

function pageFromHash(): Page {
  if (window.location.hash === "#/comparison") return "comparison";
  if (window.location.hash === "#/prepared-video") return "prepared-video";
  return "live";
}

function App() {
  const [page, setPage] = useState<Page>(pageFromHash);

  useEffect(() => {
    const updatePage = () => setPage(pageFromHash());
    window.addEventListener("hashchange", updatePage);
    return () => window.removeEventListener("hashchange", updatePage);
  }, []);

  if (page === "comparison") return <ComparisonPage />;
  if (page === "prepared-video") return <PreparedVideoPage />;
  return <LiveFusionPage />;
}

export default App;
