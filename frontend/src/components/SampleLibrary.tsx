import type { FilterName, SampleSummary } from "../types";
import { PackageUploader } from "./PackageUploader";

const FILTERS: FilterName[] = ["All", "No Crossing", "Crossing", "Irrelevant"];

interface SampleLibraryProps {
  samples: SampleSummary[];
  selectedId: string;
  filter: FilterName;
  onFilterChange: (filter: FilterName) => void;
  onSelect: (id: string) => void;
  onUpload: (file: File) => Promise<void>;
  uploading: boolean;
}

function classToken(className: string) {
  return className.toLowerCase().replace(" ", "-");
}

export function SampleLibrary({
  samples,
  selectedId,
  filter,
  onFilterChange,
  onSelect,
  onUpload,
  uploading,
}: SampleLibraryProps) {
  const visible = samples.filter(
    (sample) => filter === "All" || sample.class_name === filter,
  );

  return (
    <aside className="sample-library" aria-label="Prepared test examples">
      <div className="library-heading">
        <div>
          <p className="eyebrow">Test collection</p>
          <h2>Sequences</h2>
        </div>
        <span className="count-badge">{visible.length}</span>
      </div>

      <PackageUploader onUpload={onUpload} uploading={uploading} />

      <div className="filter-row" role="group" aria-label="Filter examples">
        {FILTERS.map((item) => (
          <button
            className={filter === item ? "active" : ""}
            key={item}
            onClick={() => onFilterChange(item)}
            type="button"
          >
            {item === "All" ? "All" : item.split(" ")[0]}
          </button>
        ))}
      </div>

      <div className="sample-list">
        {visible.map((sample) => (
          <button
            aria-pressed={selectedId === sample.id}
            className={`sample-card ${selectedId === sample.id ? "selected" : ""}`}
            key={sample.id}
            onClick={() => onSelect(sample.id)}
            type="button"
          >
            <img src={sample.poster_url} alt="" />
            <span className="sample-card-copy">
              <span className="sample-name">{sample.display_name}</span>
              <span className="sample-source">
                {sample.video_id.replace("video_", "Video ")} ·{" "}
                {sample.pedestrian_count > 1
                  ? `${sample.pedestrian_count} pedestrians`
                  : `Ped ${sample.pedestrian_id}`}
              </span>
              <span className={`class-pill ${classToken(sample.class_name)}`}>
                {sample.class_name}
              </span>
              {sample.pedestrian_count > 1 && (
                <span className="parallel-pill">Parallel</span>
              )}
              {sample.source === "upload" && (
                <span className="upload-pill">Uploaded</span>
              )}
            </span>
            <span className="sample-arrow" aria-hidden="true">
              ›
            </span>
          </button>
        ))}
      </div>
    </aside>
  );
}
