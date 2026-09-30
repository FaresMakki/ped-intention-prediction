import { usePlayback } from "../hooks/usePlayback";
import type { SampleDetail } from "../types";

interface SequenceViewerProps {
  sample: SampleDetail | null;
  loading: boolean;
}

export function SequenceViewer({ sample, loading }: SequenceViewerProps) {
  const playback = usePlayback(
    sample?.frames.length ?? 0,
    sample?.id ?? "empty",
  );
  const frame = sample?.frames[playback.frameIndex];

  return (
    <section className="viewer-card panel">
      <div className="viewer-toolbar">
        <div>
          <p className="eyebrow">Observation window</p>
          <h2>{sample ? sample.display_name : "Select an example"}</h2>
        </div>
        {sample && (
          <div className="viewer-tags">
            <span>{sample.set_id.toUpperCase()}</span>
            <span>{sample.video_id.replace("_", " ").toUpperCase()}</span>
            <span>15 frames</span>
          </div>
        )}
      </div>

      <div className={`frame-stage ${loading ? "is-loading" : ""}`}>
        {frame ? (
          <img
            key={frame.url}
            src={frame.url}
            alt={`PIE frame ${frame.source_frame}, target pedestrian ${sample?.pedestrian_id}`}
          />
        ) : (
          <div className="viewer-placeholder">Loading prepared sequence…</div>
        )}
        {frame && (
          <>
            <div className="frame-index">
              <span>{String(frame.ordinal).padStart(2, "0")}</span> / 15
            </div>
            <div className="target-legend">
              {sample?.pedestrians.map((pedestrian, index) => (
                <span className="target-person" key={pedestrian.pedestrian_id}>
                  <i style={{ background: pedestrian.color }} />P{index + 1}{" "}
                  {pedestrian.pedestrian_id}
                </span>
              ))}
            </div>
          </>
        )}
      </div>

      <div className="playback-controls">
        <button
          onClick={playback.previous}
          type="button"
          aria-label="Previous frame"
        >
          ‹
        </button>
        <button
          className="play-button"
          onClick={() => playback.setIsPlaying(!playback.isPlaying)}
          type="button"
          aria-label={playback.isPlaying ? "Pause sequence" : "Play sequence"}
        >
          {playback.isPlaying ? "Ⅱ" : "▶"}
        </button>
        <button onClick={playback.next} type="button" aria-label="Next frame">
          ›
        </button>
        <input
          aria-label="Sequence frame"
          max={Math.max((sample?.frames.length ?? 1) - 1, 0)}
          min="0"
          onChange={(event) =>
            playback.setFrameIndex(Number(event.target.value))
          }
          type="range"
          value={playback.frameIndex}
        />
        <span className="source-frame">
          Source #{frame?.source_frame ?? "—"}
        </span>
      </div>
    </section>
  );
}
