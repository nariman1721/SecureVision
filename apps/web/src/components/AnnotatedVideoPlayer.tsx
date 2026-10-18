"use client";

import {
  useMemo,
  useRef,
  useState,
} from "react";

import type {
  WorkplaceResult,
  WorkplaceTimelineItem,
} from "@/lib/types";

interface AnnotatedVideoPlayerProps {
  src: string;
  fileName: string;
  workplace: WorkplaceResult;
}

const GREEN = "#22c55e";
const RED = "#ef4444";
const YELLOW = "#facc15";

function formatTime(seconds: number): string {
  const safe = Math.max(0, seconds);
  const minutes = Math.floor(safe / 60);
  const remaining = safe % 60;

  return `${String(minutes).padStart(2, "0")}:${remaining
    .toFixed(1)
    .padStart(4, "0")}`;
}

function findSampleIndex(
  timeline: WorkplaceTimelineItem[],
  time: number,
): number {
  if (timeline.length === 0) {
    return -1;
  }

  let left = 0;
  let right = timeline.length - 1;
  let result = -1;

  while (left <= right) {
    const middle = Math.floor((left + right) / 2);

    if (timeline[middle].timecode_sec <= time + 0.001) {
      result = middle;
      left = middle + 1;
    } else {
      right = middle - 1;
    }
  }

  return result;
}

function postureLabel(posture?: string): string {
  switch (posture) {
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

function presenceLabel(state?: string): string {
  switch (state) {
    case "present":
      return "В рабочей зоне";
    case "away":
      return "Рабочая зона свободна";
    default:
      return "Не определено";
  }
}

export default function AnnotatedVideoPlayer({
  src,
  fileName,
  workplace,
}: AnnotatedVideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  const [currentTime, setCurrentTime] = useState(0);
  const [activeIndex, setActiveIndex] = useState(0);
  const [nativeSize, setNativeSize] = useState<{
    width: number;
    height: number;
  } | null>(null);

  const [showBoxes, setShowBoxes] = useState(true);
  const [showZone, setShowZone] = useState(true);
  const [showTrackLabels, setShowTrackLabels] = useState(true);
  const [showCenters, setShowCenters] = useState(true);

  const timeline = useMemo(
    () =>
      [...(workplace.timeline ?? [])].sort(
        (a, b) => a.timecode_sec - b.timecode_sec,
      ),
    [workplace.timeline],
  );

  const width = workplace.video?.width ?? 0;
  const height = workplace.video?.height ?? 0;
  const zone = workplace.configuration?.cashier_zone;

  const validDimensions = width > 0 && height > 0;

  const dimensionsMatch =
    nativeSize === null ||
    (nativeSize.width === width && nativeSize.height === height);

  const canDraw = validDimensions && dimensionsMatch;

  const sample = timeline[activeIndex] ?? null;
  const people = sample?.people ?? [];

  const inZoneCount = people.filter(
    (person) => person.in_cashier_zone,
  ).length;

  const outZoneCount = people.length - inZoneCount;

  const duration =
    workplace.video?.duration_sec ??
    videoRef.current?.duration ??
    0;

  const sampleInterval =
    workplace.visualization?.sample_interval_sec ??
    workplace.configuration?.sample_interval_sec ??
    0.5;

  function syncPlayback() {
    const video = videoRef.current;

    if (!video) {
      return;
    }

    const time = video.currentTime;

    setCurrentTime(time);

    const index = findSampleIndex(timeline, time);

    setActiveIndex(index);
  }

  function seekTo(seconds: number) {
    const video = videoRef.current;

    if (!video) {
      return;
    }

    video.currentTime = seconds;

    syncPlayback();
  }

  const quickTimes = [0, 5, 10, 12, 15, 18, 22].filter(
    (time) => time <= duration,
  );

  return (
    <section className="annotated-player-card">
      <div className="annotated-header">
        <div>
          <div className="annotated-kicker">
            WORKPLACE VISUAL MONITORING
          </div>

          <h3>Видео с AI-разметкой</h3>

          <p>
            Обнаруженные люди, анонимные Track ID и границы
            рабочей зоны поверх оригинального видео.
          </p>
        </div>

        <div className="annotated-file-badge">
          {fileName}
        </div>
      </div>

      <div className="annotated-layout">
        <div className="annotated-video-column">
          <div
            className="annotated-stage"
            style={{
              aspectRatio: validDimensions
                ? `${width} / ${height}`
                : "9 / 16",
              maxWidth:
                validDimensions && width > height
                  ? "820px"
                  : "440px",
            }}
          >
            <video
              ref={videoRef}
              src={src}
              controls
              playsInline
              preload="metadata"
              className="annotated-video"
              onTimeUpdate={syncPlayback}
              onSeeked={syncPlayback}
              onLoadedMetadata={(event) => {
                setNativeSize({
                  width: event.currentTarget.videoWidth,
                  height: event.currentTarget.videoHeight,
                });

                syncPlayback();
              }}
            />

            {canDraw && (
              <svg
                className="annotated-svg"
                viewBox={`0 0 ${width} ${height}`}
                preserveAspectRatio="xMidYMid meet"
                aria-hidden="true"
              >
                {showZone && zone && (
                  <g>
                    <rect
                      x={zone.x1 * width}
                      y={zone.y1 * height}
                      width={(zone.x2 - zone.x1) * width}
                      height={(zone.y2 - zone.y1) * height}
                      fill={YELLOW}
                      fillOpacity={0.055}
                      stroke={YELLOW}
                      strokeWidth={3}
                    />

                    <text
                      x={zone.x1 * width + 5}
                      y={zone.y1 * height + 17}
                      fill={YELLOW}
                      fontSize={14}
                      fontWeight={800}
                    >
                      WORK ZONE
                    </text>
                  </g>
                )}

                {showBoxes &&
                  people.map((person, index) => {
                    const [x1, y1, x2, y2] = person.bbox;

                    const x = Math.max(0, Math.min(width, x1));
                    const y = Math.max(0, Math.min(height, y1));

                    const boxWidth = Math.max(
                      0,
                      Math.min(width, x2) - x,
                    );

                    const boxHeight = Math.max(
                      0,
                      Math.min(height, y2) - y,
                    );

                    if (boxWidth <= 0 || boxHeight <= 0) {
                      return null;
                    }

                    const color = person.in_cashier_zone
                      ? GREEN
                      : RED;

                    const label = `Track ${person.track_id} ${person.confidence.toFixed(
                      2,
                    )} ${person.in_cashier_zone ? "IN" : "OUT"}`;

                    const labelWidth = 160;
                    const labelHeight = 22;

                    const labelX = Math.max(
                      0,
                      Math.min(width - labelWidth, x),
                    );

                    const labelY =
                      y >= labelHeight
                        ? y - labelHeight
                        : y + 2;

                    const centerX = (x1 + x2) / 2;
                    const centerY = (y1 + y2) / 2;

                    return (
                      <g key={`${person.track_id}-${index}`}>
                        <rect
                          x={x}
                          y={y}
                          width={boxWidth}
                          height={boxHeight}
                          fill="none"
                          stroke={color}
                          strokeWidth={3}
                        />

                        {showTrackLabels && (
                          <g>
                            <rect
                              x={labelX}
                              y={labelY}
                              width={labelWidth}
                              height={labelHeight}
                              rx={4}
                              fill={color}
                              fillOpacity={0.94}
                            />

                            <text
                              x={labelX + 6}
                              y={labelY + 15}
                              fill="#ffffff"
                              fontSize={12}
                              fontWeight={750}
                            >
                              {label}
                            </text>
                          </g>
                        )}

                        {showCenters && (
                          <circle
                            cx={centerX}
                            cy={centerY}
                            r={5}
                            fill={color}
                            stroke="#ffffff"
                            strokeWidth={1.5}
                          />
                        )}
                      </g>
                    );
                  })}
              </svg>
            )}

            <div className="annotated-time-label">
              {formatTime(currentTime)}
            </div>
          </div>

          {!validDimensions && (
            <p className="annotated-warning">
              В ответе AI отсутствуют размеры видео. Рамки пока
              недоступны.
            </p>
          )}

          {validDimensions && !dimensionsMatch && (
            <p className="annotated-warning">
              Размеры видео в браузере не совпадают с размерами
              анализа AI. Разметка временно скрыта, чтобы рамки
              не отображались в неправильных местах.
            </p>
          )}

          <div className="annotated-quick-seek">
            <span>Перейти к кадру:</span>

            <div>
              {quickTimes.map((time) => (
                <button
                  key={time}
                  type="button"
                  onClick={() => seekTo(time)}
                  className={
                    Math.abs(currentTime - time) < 0.25
                      ? "annotated-seek-button selected"
                      : "annotated-seek-button"
                  }
                >
                  {formatTime(time)}
                </button>
              ))}
            </div>
          </div>
        </div>

        <aside className="annotated-sidebar">
          <div className="annotated-sidebar-block">
            <span className="annotated-side-label">
              Состояние рабочей зоны
            </span>

            <strong className="annotated-state-title">
              {presenceLabel(sample?.state)}
            </strong>

            <span className="annotated-secondary">
              Положение: {postureLabel(sample?.posture)}
            </span>
          </div>

          <div className="annotated-stats-grid">
            <div>
              <span>Обнаружено людей</span>
              <strong>{people.length}</strong>
            </div>

            <div>
              <span>Внутри зоны</span>
              <strong>{inZoneCount}</strong>
            </div>

            <div>
              <span>Вне зоны</span>
              <strong>{outZoneCount}</strong>
            </div>

            <div>
              <span>Интервал анализа</span>
              <strong>{sampleInterval} с</strong>
            </div>
          </div>

          <div className="annotated-sidebar-block">
            <span className="annotated-side-label">
              Отображение разметки
            </span>

            <label className="annotated-toggle">
              <input
                type="checkbox"
                checked={showBoxes}
                onChange={(event) =>
                  setShowBoxes(event.target.checked)
                }
              />
              <span>Рамки объектов</span>
            </label>

            <label className="annotated-toggle">
              <input
                type="checkbox"
                checked={showZone}
                onChange={(event) =>
                  setShowZone(event.target.checked)
                }
              />
              <span>Рабочая зона</span>
            </label>

            <label className="annotated-toggle">
              <input
                type="checkbox"
                checked={showTrackLabels}
                disabled={!showBoxes}
                onChange={(event) =>
                  setShowTrackLabels(event.target.checked)
                }
              />
              <span>Track ID и уверенность</span>
            </label>

            <label className="annotated-toggle">
              <input
                type="checkbox"
                checked={showCenters}
                disabled={!showBoxes}
                onChange={(event) =>
                  setShowCenters(event.target.checked)
                }
              />
              <span>Центры объектов</span>
            </label>
          </div>

          <div className="annotated-sidebar-block">
            <span className="annotated-side-label">
              Объекты на текущем кадре
            </span>

            {people.length === 0 ? (
              <p className="annotated-empty">
                На данном временном отсчёте обнаруженные объекты
                отсутствуют.
              </p>
            ) : (
              <div className="annotated-person-list">
                {people.map((person, index) => (
                  <div
                    key={`${person.track_id}-${index}`}
                    className="annotated-person"
                  >
                    <span
                      className="annotated-person-dot"
                      style={{
                        background: person.in_cashier_zone
                          ? GREEN
                          : RED,
                      }}
                    />

                    <div>
                      <strong>Track {person.track_id}</strong>
                      <small>
                        {person.in_cashier_zone
                          ? "Внутри зоны"
                          : "Вне зоны"}
                      </small>
                    </div>

                    <span className="annotated-person-confidence">
                      {(person.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="annotated-legend">
            <div>
              <i style={{ background: GREEN }} />
              Внутри зоны
            </div>

            <div>
              <i style={{ background: RED }} />
              Вне зоны
            </div>

            <div>
              <i style={{ background: YELLOW }} />
              Рабочая зона
            </div>
          </div>
        </aside>
      </div>

      <p className="annotated-disclaimer">
        Рамки соответствуют проанализированным отсчётам видео
        с интервалом {sampleInterval} с. Между ними отображается
        последняя известная разметка. Track ID анонимны и не
        означают идентификацию конкретного человека.
      </p>
    </section>
  );
}