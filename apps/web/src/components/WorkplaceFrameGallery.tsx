"use client";

/* eslint-disable @next/next/no-img-element */

import {
  useEffect,
  useMemo,
  useState,
} from "react";

import type {
  WorkplaceResult,
  WorkplaceTimelineItem,
} from "@/lib/types";

interface WorkplaceFrameGalleryProps {
  src: string;
  fileName: string;
  workplace: WorkplaceResult;
}

interface GalleryFrame {
  timeSec: number;
  sampleTimeSec: number | null;
  originalSrc: string;
  annotatedSrc: string;
  peopleCount: number;
  posture: string;
  state: string;
}

const FRAME_TIMES = [
  0,
  5,
  10,
  12,
  15,
  18,
  22,
];

const COLORS = {
  inside: "#22c55e",
  outside: "#ef4444",
  zone: "#facc15",
};

function formatTime(seconds: number): string {
  const safe = Math.max(0, seconds);
  const minutes = Math.floor(safe / 60);
  const remaining = safe % 60;

  return `${String(minutes).padStart(2, "0")}:${remaining
    .toFixed(1)
    .padStart(4, "0")}`;
}

function postureLabel(value: string): string {
  switch (value) {
    case "sitting":
      return "Сидя";
    case "standing":
      return "Стоя";
    case "away":
      return "Отсутствует";
    default:
      return "Не определено";
  }
}

function findNearestSample(
  timeline: WorkplaceTimelineItem[],
  time: number,
  interval: number,
): WorkplaceTimelineItem | null {
  if (timeline.length === 0) {
    return null;
  }

  let nearest = timeline[0];
  let nearestDifference = Math.abs(
    nearest.timecode_sec - time,
  );

  for (const sample of timeline) {
    const difference = Math.abs(
      sample.timecode_sec - time,
    );

    if (difference < nearestDifference) {
      nearest = sample;
      nearestDifference = difference;
    }
  }

  if (nearestDifference > Math.max(0.3, interval)) {
    return null;
  }

  return nearest;
}

function waitForVideoReady(
  video: HTMLVideoElement,
  signal: AbortSignal,
): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new Error("Операция отменена"));
      return;
    }

    if (video.readyState >= 2) {
      resolve();
      return;
    }

    let timer: number;

    function cleanup() {
      window.clearTimeout(timer);
      video.removeEventListener("loadeddata", onReady);
      video.removeEventListener("error", onError);
      signal.removeEventListener("abort", onAbort);
    }

    function onReady() {
      cleanup();
      resolve();
    }

    function onError() {
      cleanup();
      reject(
        new Error(
          "Браузер не смог декодировать MOV-видео для создания кадров.",
        ),
      );
    }

    function onAbort() {
      cleanup();
      reject(new Error("Операция отменена"));
    }

    timer = window.setTimeout(() => {
      cleanup();
      reject(
        new Error(
          "Превышено время ожидания загрузки видео.",
        ),
      );
    }, 20000);

    video.addEventListener("loadeddata", onReady);
    video.addEventListener("error", onError);
    signal.addEventListener("abort", onAbort);
  });
}

function seekVideo(
  video: HTMLVideoElement,
  time: number,
  signal: AbortSignal,
): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new Error("Операция отменена"));
      return;
    }

    if (
      Math.abs(video.currentTime - time) < 0.005 &&
      video.readyState >= 2
    ) {
      resolve();
      return;
    }

    let timer: number;

    function cleanup() {
      window.clearTimeout(timer);
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("error", onError);
      signal.removeEventListener("abort", onAbort);
    }

    function onSeeked() {
      cleanup();
      resolve();
    }

    function onError() {
      cleanup();
      reject(
        new Error(
          `Не удалось извлечь кадр на ${formatTime(time)}.`,
        ),
      );
    }

    function onAbort() {
      cleanup();
      reject(new Error("Операция отменена"));
    }

    timer = window.setTimeout(() => {
      cleanup();
      reject(
        new Error(
          `Превышено время ожидания кадра ${formatTime(time)}.`,
        ),
      );
    }, 15000);

    video.addEventListener("seeked", onSeeked);
    video.addEventListener("error", onError);
    signal.addEventListener("abort", onAbort);

    try {
      video.currentTime = time;
    } catch {
      cleanup();
      reject(
        new Error(
          `Браузер не поддерживает переход к ${formatTime(time)}.`,
        ),
      );
    }
  });
}

function drawAnnotations(
  context: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  sample: WorkplaceTimelineItem | null,
  workplace: WorkplaceResult,
) {
  const width = canvas.width;
  const height = canvas.height;

  const sourceWidth = workplace.video?.width ?? width;
  const sourceHeight = workplace.video?.height ?? height;

  const scaleX = width / sourceWidth;
  const scaleY = height / sourceHeight;

  const zone = workplace.configuration?.cashier_zone;

  context.save();

  if (zone) {
    const x = zone.x1 * width;
    const y = zone.y1 * height;
    const zoneWidth = (zone.x2 - zone.x1) * width;
    const zoneHeight = (zone.y2 - zone.y1) * height;

    context.fillStyle = "rgba(250,204,21,0.045)";
    context.fillRect(x, y, zoneWidth, zoneHeight);

    context.strokeStyle = COLORS.zone;
    context.lineWidth = 3;
    context.strokeRect(x, y, zoneWidth, zoneHeight);

    context.font = "bold 15px Arial";
    context.textBaseline = "middle";

    const label = "WORK ZONE";
    const labelWidth = context.measureText(label).width + 14;
    const labelY = Math.min(height - 25, y + 4);

    context.fillStyle = "rgba(15,23,42,0.82)";
    context.fillRect(x, labelY, labelWidth, 23);

    context.fillStyle = COLORS.zone;
    context.fillText(label, x + 7, labelY + 11.5);
  }

  const people = sample?.people ?? [];

  for (const person of people) {
    const [x1, y1, x2, y2] = person.bbox;

    const x = x1 * scaleX;
    const y = y1 * scaleY;
    const boxWidth = (x2 - x1) * scaleX;
    const boxHeight = (y2 - y1) * scaleY;

    if (boxWidth <= 0 || boxHeight <= 0) {
      continue;
    }

    const color = person.in_cashier_zone
      ? COLORS.inside
      : COLORS.outside;

    context.strokeStyle = color;
    context.lineWidth = 3;
    context.strokeRect(x, y, boxWidth, boxHeight);

    const centerX = x + boxWidth / 2;
    const centerY = y + boxHeight / 2;

    context.beginPath();
    context.arc(centerX, centerY, 5, 0, Math.PI * 2);
    context.fillStyle = color;
    context.fill();

    context.strokeStyle = "#ffffff";
    context.lineWidth = 1;
    context.stroke();

    const label =
      `Track ${person.track_id} ` +
      `${person.confidence.toFixed(2)} ` +
      `${person.in_cashier_zone ? "IN" : "OUT"}`;

    context.font = "bold 13px Arial";
    context.textBaseline = "middle";

    const labelWidth = Math.min(
      width,
      context.measureText(label).width + 12,
    );

    const labelX = Math.max(
      0,
      Math.min(width - labelWidth, x),
    );

    const labelY = y >= 25 ? y - 25 : y + 2;

    context.fillStyle = "rgba(15,23,42,0.9)";
    context.fillRect(labelX, labelY, labelWidth, 23);

    context.fillStyle = color;
    context.fillText(label, labelX + 6, labelY + 11.5);
  }

  context.fillStyle = "rgba(15,23,42,0.72)";
  context.fillRect(8, 8, 128, 27);

  context.font = "bold 15px Arial";
  context.textBaseline = "middle";
  context.fillStyle = "#ffffff";

  context.fillText(
    `Time: ${sample?.timecode_sec.toFixed(1) ?? "—"}s`,
    14,
    21.5,
  );

  context.restore();
}

export default function WorkplaceFrameGallery({
  src,
  fileName,
  workplace,
}: WorkplaceFrameGalleryProps) {
  const [frames, setFrames] = useState<GalleryFrame[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  const [showAnnotations, setShowAnnotations] = useState(true);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(
    null,
  );
  const [retryKey, setRetryKey] = useState(0);

  const timeline = useMemo(
    () =>
      [...(workplace.timeline ?? [])].sort(
        (a, b) => a.timecode_sec - b.timecode_sec,
      ),
    [workplace.timeline],
  );

  const interval =
    workplace.visualization?.sample_interval_sec ?? 0.5;

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;

    const video = document.createElement("video");

    video.preload = "auto";
    video.muted = true;
    video.playsInline = true;

    async function generateFrames() {
      setFrames([]);
      setLoading(true);
      setError(null);
      setProgress(0);

      try {
        const readyPromise = waitForVideoReady(video, signal);

        video.src = src;
        video.load();

        await readyPromise;

        if (signal.aborted) {
          return;
        }

        const nativeWidth = video.videoWidth;
        const nativeHeight = video.videoHeight;

        if (!nativeWidth || !nativeHeight) {
          throw new Error(
            "Не удалось определить размеры видеозаписи.",
          );
        }

        const canvas = document.createElement("canvas");

        const resize = Math.min(
          1,
          1100 / Math.max(nativeWidth, nativeHeight),
        );

        canvas.width = Math.round(nativeWidth * resize);
        canvas.height = Math.round(nativeHeight * resize);

        const context = canvas.getContext("2d");

        if (!context) {
          throw new Error(
            "Не удалось создать Canvas для обработки кадров.",
          );
        }

        const duration = Number.isFinite(video.duration)
          ? video.duration
          : workplace.video?.duration_sec ?? 0;

        if (duration <= 0) {
          throw new Error(
            "Длительность видео не определена.",
          );
        }

        const metadataWidth = workplace.video?.width;
        const metadataHeight = workplace.video?.height;

        const dimensionsMatch =
          (!metadataWidth || metadataWidth === nativeWidth) &&
          (!metadataHeight || metadataHeight === nativeHeight);

        if (!dimensionsMatch) {
          throw new Error(
            "Размеры видео отличаются от размеров AI-анализа. " +
            "Для правильной разметки необходимо согласовать ориентацию кадров.",
          );
        }

        const timestamps = FRAME_TIMES.filter(
          (time) => time < duration,
        );

        const generated: GalleryFrame[] = [];

        for (let index = 0; index < timestamps.length; index++) {
          if (signal.aborted) {
            return;
          }

          const time = timestamps[index];

          await seekVideo(
            video,
            time,
            signal,
          );

          if (signal.aborted) {
            return;
          }

          context.drawImage(
            video,
            0,
            0,
            canvas.width,
            canvas.height,
          );

          const originalSrc = canvas.toDataURL(
            "image/jpeg",
            0.86,
          );

          const sample = findNearestSample(
            timeline,
            time,
            interval,
          );

          drawAnnotations(
            context,
            canvas,
            sample,
            workplace,
          );

          const annotatedSrc = canvas.toDataURL(
            "image/jpeg",
            0.88,
          );

          generated.push({
            timeSec: time,
            sampleTimeSec: sample?.timecode_sec ?? null,
            originalSrc,
            annotatedSrc,
            peopleCount: sample?.people?.length ?? 0,
            posture: sample?.posture ?? "unknown",
            state: sample?.state ?? "unknown",
          });

          if (!signal.aborted) {
            setProgress(index + 1);
          }
        }

        if (!signal.aborted) {
          setFrames(generated);
          setLoading(false);
        }
      } catch (unknownError) {
        if (!signal.aborted) {
          setError(
            unknownError instanceof Error
              ? unknownError.message
              : "Не удалось создать галерею кадров.",
          );

          setLoading(false);
        }
      }
    }

    void generateFrames();

    return () => {
      controller.abort();
      video.pause();
      video.removeAttribute("src");
      video.load();
    };
  }, [src, workplace, timeline, interval, retryKey]);

  useEffect(() => {
    if (selectedIndex === null) {
      return;
    }

    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setSelectedIndex(null);
      }

      if (event.key === "ArrowRight") {
        setSelectedIndex((current) =>
          current === null
            ? null
            : Math.min(frames.length - 1, current + 1),
        );
      }

      if (event.key === "ArrowLeft") {
        setSelectedIndex((current) =>
          current === null
            ? null
            : Math.max(0, current - 1),
        );
      }
    }

    window.addEventListener("keydown", onKeyDown);

    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [selectedIndex, frames.length]);

  function downloadFrame(frame: GalleryFrame) {
    const image = showAnnotations
      ? frame.annotatedSrc
      : frame.originalSrc;

    const fileBase = fileName.replace(/\.mov$/i, "");

    const anchor = document.createElement("a");

    anchor.href = image;
    anchor.download =
      `${fileBase}-${frame.timeSec.toFixed(1)}s-` +
      `${showAnnotations ? "annotated" : "original"}.jpg`;

    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
  }

  const selectedFrame =
    selectedIndex === null
      ? null
      : frames[selectedIndex] ?? null;

  return (
    <section className="frame-gallery-card">
      <div className="frame-gallery-header">
        <div>
          <span className="frame-gallery-kicker">
            AI DETECTION FRAMES
          </span>

          <h3>Галерея размеченных кадров</h3>

          <p>
            Реальные кадры из {fileName} с результатами
            обнаружения людей и рабочей зоны.
          </p>
        </div>

        <div className="frame-gallery-controls">
          <button
            type="button"
            className={
              showAnnotations
                ? "frame-gallery-control active"
                : "frame-gallery-control"
            }
            onClick={() => setShowAnnotations(true)}
          >
            С AI-разметкой
          </button>

          <button
            type="button"
            className={
              !showAnnotations
                ? "frame-gallery-control active"
                : "frame-gallery-control"
            }
            onClick={() => setShowAnnotations(false)}
          >
            Оригинал
          </button>
        </div>
      </div>

      {loading && (
        <div className="frame-gallery-loading">
          <div className="spinner" />

          <div>
            <strong>Создаём галерею кадров</strong>
            <p>
              Обработано {progress} из{" "}
              {FRAME_TIMES.length} временных отметок.
            </p>
          </div>
        </div>
      )}

      {error && (
        <div className="frame-gallery-error">
          <strong>Не удалось создать галерею</strong>

          <p>{error}</p>

          <button
            type="button"
            onClick={() => setRetryKey((current) => current + 1)}
          >
            Повторить
          </button>
        </div>
      )}

      {!loading && !error && frames.length > 0 && (
        <>
          <div className="frame-gallery-grid">
            {frames.map((frame, index) => (
              <button
                type="button"
                key={frame.timeSec}
                className="frame-gallery-item"
                onClick={() => setSelectedIndex(index)}
              >
                <div className="frame-gallery-image">
                  <img
                    src={
                      showAnnotations
                        ? frame.annotatedSrc
                        : frame.originalSrc
                    }
                    alt={`Кадр видео на ${formatTime(frame.timeSec)}`}
                  />

                  <span className="frame-gallery-time">
                    {formatTime(frame.timeSec)}
                  </span>
                </div>

                <div className="frame-gallery-item-info">
                  <strong>{postureLabel(frame.posture)}</strong>

                  <span>
                    Людей: {frame.peopleCount}
                  </span>
                </div>
              </button>
            ))}
          </div>

          <p className="frame-gallery-note">
            Кадры извлечены из оригинального видео в браузере.
            Рамки построены на основе уже рассчитанных Python
            detections. Изображения не отправляются в облако.
            Между отсчётами 0.5 секунды отдельные измерения
            положения людей не выполнялись.
          </p>
        </>
      )}

      {selectedFrame && selectedIndex !== null && (
        <div
          className="frame-gallery-modal"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              setSelectedIndex(null);
            }
          }}
        >
          <div
            className="frame-gallery-dialog"
            role="dialog"
            aria-modal="true"
            aria-label={`Кадр ${formatTime(selectedFrame.timeSec)}`}
          >
            <div className="frame-gallery-modal-header">
              <div>
                <strong>
                  Кадр {formatTime(selectedFrame.timeSec)}
                </strong>

                <span>
                  {postureLabel(selectedFrame.posture)}
                  {" · "}
                  Объектов: {selectedFrame.peopleCount}
                </span>
              </div>

              <button
                type="button"
                className="frame-gallery-close"
                onClick={() => setSelectedIndex(null)}
                aria-label="Закрыть изображение"
              >
                ×
              </button>
            </div>

            <div className="frame-gallery-modal-image">
              <img
                src={
                  showAnnotations
                    ? selectedFrame.annotatedSrc
                    : selectedFrame.originalSrc
                }
                alt={`Увеличенный кадр ${formatTime(
                  selectedFrame.timeSec,
                )}`}
              />
            </div>

            <div className="frame-gallery-modal-actions">
              <button
                type="button"
                disabled={selectedIndex === 0}
                onClick={() =>
                  setSelectedIndex((current) =>
                    current === null ? null : Math.max(0, current - 1),
                  )
                }
              >
                ← Предыдущий
              </button>

              <button
                type="button"
                onClick={() => downloadFrame(selectedFrame)}
              >
                ↓ Скачать JPEG
              </button>

              <button
                type="button"
                disabled={selectedIndex === frames.length - 1}
                onClick={() =>
                  setSelectedIndex((current) =>
                    current === null
                      ? null
                      : Math.min(frames.length - 1, current + 1),
                  )
                }
              >
                Следующий →
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}