import { useCallback, useEffect, useState } from "react";

export function usePlayback(frameCount: number, sequenceKey: string) {
  const [frameIndex, setFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);

  useEffect(() => {
    setFrameIndex(0);
    setIsPlaying(true);
  }, [sequenceKey]);

  useEffect(() => {
    if (!isPlaying || frameCount < 2) return;
    const timer = window.setInterval(() => {
      setFrameIndex((current) => (current + 1) % frameCount);
    }, 170);
    return () => window.clearInterval(timer);
  }, [frameCount, isPlaying]);

  const previous = useCallback(() => {
    setFrameIndex((current) => (current - 1 + frameCount) % frameCount);
  }, [frameCount]);

  const next = useCallback(() => {
    setFrameIndex((current) => (current + 1) % frameCount);
  }, [frameCount]);

  return {
    frameIndex,
    isPlaying,
    setFrameIndex,
    setIsPlaying,
    previous,
    next,
  };
}
