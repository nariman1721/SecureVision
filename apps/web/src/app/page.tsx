"use client";

import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import type {
  ChangeEvent,
  DragEvent,
} from "react";

import {
  submitFullAnalysis,
  waitForFullAnalysis,
} from "@/lib/api";

import type {
  EvidenceFrame,
  FullAnalysisJob,
  Violation,
  WorkplaceTimelineItem,
} from "@/lib/types";
import AnnotatedVideoPlayer from "@/components/AnnotatedVideoPlayer";
import WorkplaceFrameGallery from "@/components/WorkplaceFrameGallery";

const MAX_FILE_SIZE =
  150 * 1024 * 1024;


const LEGAL_SOURCE = {
  title:
    "Постановление Правления Национального Банка Республики Казахстан от 29 ноября 2019 года № 231",

  registration:
    "Зарегистрировано в Министерстве юстиции РК № 19680",

  url:
    "https://adilet.zan.kz/rus/docs/V1900019680",
};


type ActivityState =
  | "sitting"
  | "standing"
  | "away"
  | "unknown";


interface ActivitySegment {
  state: ActivityState;

  startSec: number;
  endSec: number;
  durationSec: number;
}


function formatBytes(
  bytes: number,
) {
  if (bytes === 0) {
    return "0 MB";
  }

  const mb =
    bytes / 1024 / 1024;

  return `${mb.toFixed(1)} MB`;
}


function formatSeconds(
  value:
    number
    | null
    | undefined,
) {
  if (
    value === null ||
    value === undefined
  ) {
    return "—";
  }

  return `${value.toFixed(1)} сек`;
}


function formatTimecode(
  seconds:
    number
    | null
    | undefined,
) {
  if (
    seconds === null ||
    seconds === undefined
  ) {
    return "—";
  }

  const minutes =
    Math.floor(
      seconds / 60,
    );

  const remaining =
    seconds -
    minutes * 60;

  return `${String(
    minutes,
  ).padStart(
    2,
    "0",
  )}:${remaining
    .toFixed(1)
    .padStart(
      4,
      "0",
    )}`;
}


function extractBase64(
  frame:
    EvidenceFrame,
): string | null {
  if (
    typeof frame ===
    "string"
  ) {
    return frame;
  }

  return (
    frame.base64 ??
    frame.image_base64 ??
    frame.jpeg_base64 ??
    frame.frame_base64 ??
    null
  );
}


function getEvidence(
  violation:
    Violation,
) {
  const frames =
    violation.evidence_frames ??
    violation.evidence ??
    [];

  return frames
    .map(
      extractBase64,
    )
    .filter(
      (
        value,
      ): value is string =>
        Boolean(
          value,
        ),
    );
}


function translateViolationDescription(
  violation:
    Violation,
) {
  if (
    violation.type ===
    "suspected_cash_discipline_violation"
  ) {
    return (
      "ИИ выявил визуальный паттерн, " +
      "связанный с возможным нарушением " +
      "кассовой дисциплины. Обучающая " +
      "выборка содержит бинарные метки, " +
      "поэтому система не определяет " +
      "конкретную нарушенную правовую " +
      "норму автоматически."
    );
  }

  return (
    violation.description ??
    "Обнаружено подозрительное событие, требующее проверки специалистом."
  );
}


function getActivityState(
  item:
    WorkplaceTimelineItem,
): ActivityState {
  if (
    item.state === "away" ||
    item.posture === "away"
  ) {
    return "away";
  }

  if (
    item.posture ===
    "sitting"
  ) {
    return "sitting";
  }

  if (
    item.posture ===
    "standing"
  ) {
    return "standing";
  }

  return "unknown";
}


function buildActivitySegments(
  timeline:
    WorkplaceTimelineItem[],
): ActivitySegment[] {
  const segments:
    ActivitySegment[] = [];

  for (
    const item
    of timeline
  ) {
    const state =
      getActivityState(
        item,
      );

    const startSec =
      item.timecode_sec;

    const durationSec =
      Math.max(
        0,
        item.sample_duration_sec,
      );

    const endSec =
      startSec +
      durationSec;

    const previous =
      segments[
        segments.length - 1
      ];

    if (
      previous &&
      previous.state === state &&
      Math.abs(
        previous.endSec -
        startSec,
      ) < 0.06
    ) {
      previous.endSec =
        endSec;

      previous.durationSec +=
        durationSec;

      continue;
    }

    segments.push({
      state,
      startSec,
      endSec,
      durationSec,
    });
  }

  return segments;
}


function activityLabel(
  state:
    ActivityState,
) {
  switch (state) {
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


export default function Home() {
  const inputRef =
    useRef<HTMLInputElement>(
      null,
    );

  const [
    selectedFile,
    setSelectedFile,
  ] =
    useState<File | null>(
      null,
    );

  const [
    previewUrl,
    setPreviewUrl,
  ] =
    useState<string | null>(
      null,
    );

  const [
    isDragging,
    setIsDragging,
  ] =
    useState(false);

  const [
    isAnalyzing,
    setIsAnalyzing,
  ] =
    useState(false);

  const [
    job,
    setJob,
  ] =
    useState<FullAnalysisJob | null>(
      null,
    );

  const [
    error,
    setError,
  ] =
    useState<string | null>(
      null,
    );


  useEffect(
    () => {
      if (
        !selectedFile
      ) {
        setPreviewUrl(
          null,
        );

        return;
      }

      const url =
        URL.createObjectURL(
          selectedFile,
        );

      setPreviewUrl(
        url,
      );

      return () => {
        URL.revokeObjectURL(
          url,
        );
      };
    },
    [
      selectedFile,
    ],
  );


  const result =
    job?.result ??
    null;


  const cashSummary =
    result?.summary
      .cash_discipline;


  /*
   * В summary unified API находится сокращённая версия.
   * В workplace.summary находится полный набор метрик.
   */
  const workplaceSummary =
    result?.workplace
      .summary ??
    result?.summary
      .workplace;


  const violations =
    result
      ?.cash_discipline
      .violations ??
    [];


  /*
   * Используем browser File.name первым.
   * Поэтому UI показывает X-18.mov,
   * а не временное input.mov.
   */
  const displayFileName =
    selectedFile?.name ??
    job?.file_name ??
    result?.file_name ??
    "video.mov";


  const activitySegments =
    useMemo(
      () =>
        buildActivitySegments(
          result?.workplace
            .timeline ??
            [],
        ),
      [
        result,
      ],
    );


  const activityDuration =
    useMemo(
      () =>
        activitySegments.reduce(
          (
            total,
            segment,
          ) =>
            total +
            segment.durationSec,
          0,
        ),
      [
        activitySegments,
      ],
    );


  const posturePercentages =
    useMemo(
      () => {
        const sitting =
          workplaceSummary
            ?.sitting_percent_of_known_posture ??
          0;

        const standing =
          workplaceSummary
            ?.standing_percent_of_known_posture ??
          0;

        return {
          sitting,
          standing,
        };
      },
      [
        workplaceSummary,
      ],
    );


  function validateAndSetFile(
    file:
      File,
  ) {
    setError(
      null,
    );

    const extension =
      file.name
        .split(".")
        .pop()
        ?.toLowerCase();

    if (
      extension !==
      "mov"
    ) {
      setError(
        "Можно загрузить только видео в формате .MOV.",
      );

      return;
    }

    if (
      file.size >
      MAX_FILE_SIZE
    ) {
      setError(
        "Размер видео не должен превышать 150 MB.",
      );

      return;
    }

    if (
      file.size === 0
    ) {
      setError(
        "Выбранный файл пустой.",
      );

      return;
    }

    setSelectedFile(
      file,
    );

    setJob(
      null,
    );
  }


  function handleFileChange(
    event:
      ChangeEvent<HTMLInputElement>,
  ) {
    const file =
      event.target
        .files?.[0];

    if (file) {
      validateAndSetFile(
        file,
      );
    }
  }


  function handleDrop(
    event:
      DragEvent<HTMLDivElement>,
  ) {
    event.preventDefault();

    setIsDragging(
      false,
    );

    const file =
      event.dataTransfer
        .files?.[0];

    if (file) {
      validateAndSetFile(
        file,
      );
    }
  }


  function resetAnalysis() {
    setSelectedFile(
      null,
    );

    setJob(
      null,
    );

    setError(
      null,
    );

    setIsAnalyzing(
      false,
    );

    if (
      inputRef.current
    ) {
      inputRef.current.value =
        "";
    }

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  }


  async function handleAnalyze() {
    if (
      !selectedFile
    ) {
      return;
    }

    try {
      setError(
        null,
      );

      setIsAnalyzing(
        true,
      );

      const submitted =
        await submitFullAnalysis(
          selectedFile,
        );

      setJob({
        ...submitted,
        result: null,
        error: null,
      });

      const completed =
        await waitForFullAnalysis(
          submitted.job_id,
          {
            intervalMs:
              1000,

            timeoutMs:
              180_000,

            onUpdate:
              setJob,
          },
        );

      setJob(
        completed,
      );

      window.scrollTo({
        top: 0,
        behavior: "smooth",
      });
    } catch (
      unknownError
    ) {
      setError(
        unknownError instanceof
          Error
          ? unknownError.message
          : "Не удалось выполнить анализ.",
      );
    } finally {
      setIsAnalyzing(
        false,
      );
    }
  }


  function downloadJsonReport() {
    if (
      !result
    ) {
      return;
    }

    const report = {
      ...result,

      file_name:
        displayFileName,
    };

    const blob =
      new Blob(
        [
          JSON.stringify(
            report,
            null,
            2,
          ),
        ],
        {
          type:
            "application/json",
        },
      );

    const url =
      URL.createObjectURL(
        blob,
      );

    const anchor =
      document.createElement(
        "a",
      );

    anchor.href =
      url;

    anchor.download =
      `securevision-${displayFileName.replace(
        /\.mov$/i,
        "",
      )}-report.json`;

    document.body.appendChild(
      anchor,
    );

    anchor.click();

    anchor.remove();

    URL.revokeObjectURL(
      url,
    );
  }


  const analysisDone =
    Boolean(
      result &&
      job?.status ===
        "done",
    );


  return (
    <main className="page-shell">

      <header className="topbar">

        <div className="topbar-inner">

          <div className="brand">

            <div className="brand-icon">
              F
            </div>

            <div>

              <div className="brand-name">
                SecureVision
              </div>

              <div className="brand-subtitle">
                AI Video Compliance
              </div>

            </div>

          </div>


          <div className="header-right">

            <span className="local-badge">
              ● LOCAL AI
            </span>

            <span className="demo-badge">
              DEMO
            </span>

          </div>

        </div>

      </header>


      <section className="content">

        {!analysisDone && (
          <>

            <div className="hero">

              <div className="eyebrow">
                VIDEO ANALYTICS
              </div>

              <h1>
                Проверка видео
              </h1>

              <p>
                Загрузите запись кассовой зоны,
                чтобы проверить кассовую дисциплину
                и активность сотрудника.
              </p>


              <div className="capability-row">

                <span>
                  Кассовая дисциплина
                </span>

                <span>
                  Активность сотрудника
                </span>

                <span>
                  Фотофиксация
                </span>

                <span>
                  Локальная обработка
                </span>

              </div>

            </div>


            <section className="upload-card">

              <div className="card-heading">

                <div>

                  <h2>
                    Загрузка видео
                  </h2>

                  <p>
                    Выберите запись для комплексного AI-анализа.
                  </p>

                </div>

                <div className="secure-label">
                  Без распознавания лиц
                </div>

              </div>


              {!selectedFile ? (

                <div
                  className={
                    isDragging
                      ? "dropzone dropzone-active"
                      : "dropzone"
                  }

                  onClick={() =>
                    inputRef.current?.click()
                  }

                  onDragOver={(
                    event,
                  ) => {
                    event.preventDefault();

                    setIsDragging(
                      true,
                    );
                  }}

                  onDragLeave={() =>
                    setIsDragging(
                      false,
                    )
                  }

                  onDrop={
                    handleDrop
                  }
                >

                  <input
                    ref={
                      inputRef
                    }

                    type="file"

                    accept=".mov,video/quicktime"

                    hidden

                    onChange={
                      handleFileChange
                    }
                  />


                  <div className="upload-icon">
                    ↑
                  </div>

                  <strong>
                    Нажмите или перетащите видео сюда
                  </strong>

                  <span>
                    Формат .MOV • до 150 MB
                  </span>

                </div>

              ) : (

                <div className="selected-file">

                  <div className="selected-preview">

                    {previewUrl && (
                      <video
                        src={
                          previewUrl
                        }

                        controls

                        playsInline
                      />
                    )}

                  </div>


                  <div className="file-info">

                    <div className="file-status">
                      ✓ ГОТОВО К АНАЛИЗУ
                    </div>

                    <h3>
                      {
                        selectedFile.name
                      }
                    </h3>

                    <p>
                      {formatBytes(
                        selectedFile.size,
                      )}
                      {" • "}
                      MOV Video
                    </p>

                    <button
                      type="button"

                      className="remove-button"

                      disabled={
                        isAnalyzing
                      }

                      onClick={
                        resetAnalysis
                      }
                    >
                      Выбрать другое видео
                    </button>

                  </div>

                </div>

              )}


              {error && (

                <div className="error-box">

                  <strong>
                    Ошибка
                  </strong>

                  <span>
                    {error}
                  </span>

                </div>

              )}


              {isAnalyzing && (

                <div className="processing-panel">

                  <div className="processing-heading">

                    <div className="spinner" />

                    <div>

                      <strong>
                        SecureVision анализирует видео
                      </strong>

                      <span>
                        Обработка выполняется локально
                      </span>

                    </div>

                  </div>


                  <div className="analysis-steps">

                    <div className="analysis-step done">
                      <i>
                        ✓
                      </i>

                      <span>
                        Видео загружено
                      </span>
                    </div>


                    <div className="analysis-step active">
                      <i>
                        2
                      </i>

                      <span>
                        Кассовая дисциплина
                      </span>
                    </div>


                    <div className="analysis-step active">
                      <i>
                        3
                      </i>

                      <span>
                        Workplace Analytics
                      </span>
                    </div>


                    <div className="analysis-step">
                      <i>
                        4
                      </i>

                      <span>
                        Формирование отчёта
                      </span>
                    </div>

                  </div>

                </div>

              )}


              <div className="upload-actions">

                <div className="privacy-note">
                  Видео анализируется локально
                  и не отправляется во внешние AI-сервисы.
                </div>


                <button
                  type="button"

                  className="primary-button"

                  disabled={
                    !selectedFile ||
                    isAnalyzing
                  }

                  onClick={
                    handleAnalyze
                  }
                >

                  {isAnalyzing
                    ? "Анализируем..."
                    : "Начать сканирование"}

                </button>

              </div>

            </section>

          </>
        )}



        {analysisDone &&
          result && (
            <>

              <div className="hero results-hero">

                <div className="eyebrow">
                  ANALYSIS COMPLETE
                </div>

                <h1>
                  Результаты проверки
                </h1>

                <p>
                  Комплексный анализ кассовой дисциплины
                  и активности сотрудника завершён.
                </p>


                <div className="result-file-chip">

                  <span className="result-file-icon">
                    ▶
                  </span>

                  <div>

                    <strong>
                      {
                        displayFileName
                      }
                    </strong>

                    <small>
                      Локальный AI-анализ • MOV
                    </small>

                  </div>

                </div>

              </div>



              <section
                className={
                  cashSummary
                    ?.violation_detected
                    ? "verdict-banner verdict-danger"
                    : "verdict-banner verdict-success"
                }
              >

                <div className="verdict-icon">

                  {cashSummary
                    ?.violation_detected
                    ? "!"
                    : "✓"}

                </div>


                <div>

                  <span className="verdict-label">
                    КАССОВАЯ ДИСЦИПЛИНА
                  </span>

                  <h2>

                    {cashSummary
                      ?.violation_detected
                      ? "Обнаружены подозрительные события"
                      : "Подозрительных событий не обнаружено"}

                  </h2>

                  <p>
                    Анализ завершён за{" "}
                    {result.processing.total_sec.toFixed(
                      2,
                    )}{" "}
                    сек.
                  </p>

                </div>


                {cashSummary
                  ?.violation_detected && (

                  <span className="review-badge">
                    ТРЕБУЕТ ПРОВЕРКИ
                  </span>

                )}

              </section>



              <div className="metrics-grid">

                <article className="metric-card">

                  <span>
                    Нарушения
                  </span>

                  <strong>
                    {cashSummary
                      ?.violation_count ??
                      0}
                  </strong>

                  <small>
                    подозрительных событий
                  </small>

                </article>


                <article className="metric-card">

                  <span>
                    Присутствие
                  </span>

                  <strong>
                    {(
                      workplaceSummary
                        ?.presence_percent ??
                      0
                    ).toFixed(
                      0,
                    )}
                    %
                  </strong>

                  <small>
                    рабочая зона
                  </small>

                </article>


                <article className="metric-card">

                  <span>
                    Сидя
                  </span>

                  <strong>
                    {formatSeconds(
                      workplaceSummary
                        ?.sitting_sec,
                    )}
                  </strong>

                  <small>
                    положение сотрудника
                  </small>

                </article>


                <article className="metric-card">

                  <span>
                    Стоя
                  </span>

                  <strong>
                    {formatSeconds(
                      workplaceSummary
                        ?.standing_sec,
                    )}
                  </strong>

                  <small>
                    положение сотрудника
                  </small>

                </article>

              </div>



              <div className="section-heading">

                <div>

                  <span className="section-kicker">
                    CASH DISCIPLINE
                  </span>

                  <h2>
                    Обнаруженные события
                  </h2>

                </div>


                <span className="count-badge">
                  {
                    violations.length
                  }{" "}
                  событий
                </span>

              </div>



              {violations.length ===
              0 ? (

                <section className="empty-result">

                  <div className="empty-icon">
                    ✓
                  </div>

                  <h3>
                    Подозрительные события не обнаружены
                  </h3>

                  <p>
                    SecureVision не выявил событий,
                    превышающих production threshold.
                  </p>

                </section>

              ) : (

                <div className="violation-list">

                  {violations.map(
                    (
                      violation,
                      index,
                    ) => {

                      const evidence =
                        getEvidence(
                          violation,
                        );

                      return (

                        <article
                          className="violation-card"

                          key={`${violation.timecode_sec}-${index}`}
                        >

                          <div className="violation-header">

                            <div>

                              <span className="violation-number">
                                СОБЫТИЕ #
                                {index + 1}
                              </span>

                              <h3>
                                {translateViolationDescription(
                                  violation,
                                )}
                              </h3>

                            </div>


                            <div className="timecode">
                              {formatTimecode(
                                violation.timecode_sec,
                              )}
                            </div>

                          </div>


                          <div className="violation-meta">

                            <span>
                              Категория:{" "}

                              <strong>
                                Подозрение на нарушение
                                кассовой дисциплины
                              </strong>
                            </span>


                            {typeof violation.confidence ===
                              "number" && (

                              <span>
                                Уверенность модели:{" "}

                                <strong>
                                  {(
                                    violation.confidence *
                                    100
                                  ).toFixed(
                                    1,
                                  )}
                                  %
                                </strong>
                              </span>

                            )}

                          </div>



                          {evidence.length >
                            0 && (
                            <>

                              <div className="evidence-label">
                                Кадры подтверждения
                              </div>


                              <div className="evidence-grid">

                                {evidence.map(
                                  (
                                    base64,
                                    evidenceIndex,
                                  ) => (

                                    <div
                                      className="evidence-image"

                                      key={
                                        evidenceIndex
                                      }
                                    >

                                      <img
                                        src={`data:image/jpeg;base64,${base64}`}

                                        alt={`Кадр подтверждения ${evidenceIndex + 1}`}
                                      />

                                      <span className="evidence-index">
                                        Кадр{" "}
                                        {evidenceIndex +
                                          1}
                                      </span>

                                    </div>

                                  ),
                                )}

                              </div>

                            </>
                          )}


                          <div className="legal-inline">

                            <div className="legal-icon">
                              §
                            </div>

                            <div>

                              <strong>
                                Требуется проверка специалистом
                              </strong>

                              <span>
                                Результат является автоматизированным
                                индикатором и не считается окончательным
                                юридическим заключением.
                              </span>

                            </div>

                          </div>

                        </article>

                      );
                    },
                  )}

                </div>

              )}



              <div className="section-heading legal-heading">

                <div>

                  <span className="section-kicker">
                    LEGAL CONTEXT
                  </span>

                  <h2>
                    Нормативная база
                  </h2>

                </div>

              </div>


              <section className="legal-source-card">

                <div className="legal-source-mark">
                  §
                </div>


                <div className="legal-source-content">

                  <div className="official-source-label">
                    ОФИЦИАЛЬНЫЙ ИСТОЧНИК
                  </div>

                  <h3>
                    {
                      LEGAL_SOURCE.title
                    }
                  </h3>

                  <p>
                    {
                      LEGAL_SOURCE.registration
                    }
                  </p>

                  <a
                    href={
                      LEGAL_SOURCE.url
                    }

                    target="_blank"

                    rel="noreferrer"
                  >
                    Открыть документ на Adilet.kz ↗
                  </a>

                </div>


                <div className="legal-warning">

                  <strong>
                    Human review
                  </strong>

                  <span>
                    SecureVision не выносит
                    автоматическое юридическое заключение.
                  </span>

                </div>

              </section>



              <div className="section-heading workplace-heading">

                <div>

                  <span className="section-kicker">
                    WORKPLACE ACTIVITY
                  </span>

                  <h2>
                    Активность сотрудника
                  </h2>

                </div>


                <span className="privacy-badge">
                  ANONYMOUS TRACKING
                </span>

              </div>



              <section className="workplace-card">

                <div className="workplace-stats">

                  <div>

                    <span>
                      Присутствие
                    </span>

                    <strong>
                      {(
                        workplaceSummary
                          ?.presence_percent ??
                        0
                      ).toFixed(
                        1,
                      )}
                      %
                    </strong>

                  </div>


                  <div>

                    <span>
                      Отсутствие
                    </span>

                    <strong>
                      {formatSeconds(
                        workplaceSummary
                          ?.away_sec ??
                        0,
                      )}
                    </strong>

                  </div>


                  <div>

                    <span>
                      Периоды отсутствия
                    </span>

                    <strong>
                      {workplaceSummary
                        ?.absence_count ??
                        0}
                    </strong>

                  </div>


                  <div>

                    <span>
                      Самое долгое отсутствие
                    </span>

                    <strong>

                      {(
                        workplaceSummary
                          ?.absence_count ??
                        0
                      ) === 0
                        ? "Нет"
                        : formatSeconds(
                            workplaceSummary
                              ?.longest_absence_sec,
                          )}

                    </strong>

                  </div>

                </div>



                <div className="posture-area">

                  <div className="posture-row">

                    <div className="posture-label">

                      <span>
                        Сидя
                      </span>

                      <strong>
                        {posturePercentages.sitting.toFixed(
                          1,
                        )}
                        %
                      </strong>

                    </div>


                    <div className="bar-track">

                      <div
                        className="bar-fill sitting-bar"

                        style={{
                          width:
                            `${Math.min(
                              100,
                              posturePercentages.sitting,
                            )}%`,
                        }}
                      />

                    </div>

                  </div>



                  <div className="posture-row">

                    <div className="posture-label">

                      <span>
                        Стоя
                      </span>

                      <strong>
                        {posturePercentages.standing.toFixed(
                          1,
                        )}
                        %
                      </strong>

                    </div>


                    <div className="bar-track">

                      <div
                        className="bar-fill standing-bar"

                        style={{
                          width:
                            `${Math.min(
                              100,
                              posturePercentages.standing,
                            )}%`,
                        }}
                      />

                    </div>

                  </div>

                </div>



                {activitySegments.length >
                  0 && (

                  <div className="activity-section">

                    <div className="activity-title-row">

                      <div>

                        <strong>
                          Хронология активности
                        </strong>

                        <span>
                          Изменение положения сотрудника
                          по времени
                        </span>

                      </div>


                      <span className="timeline-duration">
                        {formatTimecode(
                          activityDuration,
                        )}
                      </span>

                    </div>


                    <div className="activity-timeline">

                      {activitySegments.map(
                        (
                          segment,
                          index,
                        ) => (

                          <div
                            key={`${segment.startSec}-${index}`}

                            className={
                              `activity-segment activity-${segment.state}`
                            }

                            style={{
                              width:
                                `${
                                  activityDuration >
                                  0
                                    ? (
                                        segment.durationSec /
                                        activityDuration
                                      ) *
                                      100
                                    : 0
                                }%`,
                            }}

                            title={
                              `${activityLabel(
                                segment.state,
                              )}: ${formatTimecode(
                                segment.startSec,
                              )}–${formatTimecode(
                                segment.endSec,
                              )}`
                            }
                          />

                        ),
                      )}

                    </div>


                    <div className="timeline-axis">

                      <span>
                        00:00.0
                      </span>

                      <span>
                        {formatTimecode(
                          activityDuration /
                          2,
                        )}
                      </span>

                      <span>
                        {formatTimecode(
                          activityDuration,
                        )}
                      </span>

                    </div>


                    <div className="activity-legend">

                      <span>
                        <i className="legend-sitting" />
                        Сидя
                      </span>

                      <span>
                        <i className="legend-standing" />
                        Стоя
                      </span>

                      <span>
                        <i className="legend-away" />
                        Отсутствует
                      </span>

                      <span>
                        <i className="legend-unknown" />
                        Не определено
                      </span>

                    </div>

                  </div>

                )}



                <div className="workplace-note">

                  <strong>
                    Privacy by design
                  </strong>

                  <span>
                    Система использует анонимный tracking
                    и геометрию bounding box.
                    Распознавание лиц и биометрическая
                    идентификация не используются.
                  </span>

                </div>

              </section>



              <section className="processing-summary">
                {previewUrl && (
                  <>
                    <div className="section-heading workplace-heading">
                      <div>
                        <span className="section-kicker">
                          AI VIDEO MONITORING
                        </span>

                        <h2>
                          Визуальный анализ видео
                        </h2>
                      </div>

                      <span className="privacy-badge">
                        REAL DETECTIONS
                      </span>
                    </div>

                    <AnnotatedVideoPlayer
                      src={previewUrl}
                      fileName={displayFileName}
                      workplace={result.workplace}
                    />
                    <WorkplaceFrameGallery
                      src={previewUrl}
                      fileName={displayFileName}
                      workplace={result.workplace}
                    />
                  </>
                )}
                <div>

                  <span>
                    Cash AI
                  </span>

                  <strong>
                    {result.processing
                      .cash_discipline_sec.toFixed(
                        2,
                      )}
                    s
                  </strong>

                </div>


                <div>

                  <span>
                    Workplace AI
                  </span>

                  <strong>
                    {result.processing
                      .workplace_sec.toFixed(
                        2,
                      )}
                    s
                  </strong>

                </div>


                <div>

                  <span>
                    Общее время
                  </span>

                  <strong>
                    {result.processing
                      .total_sec.toFixed(
                        2,
                      )}
                    s
                  </strong>

                </div>

              </section>



              <details className="technical-details">

                <summary>
                  Технические детали
                </summary>

                <pre>

                  {JSON.stringify(
                    {
                      ...result,

                      file_name:
                        displayFileName,
                    },
                    null,
                    2,
                  )}

                </pre>

              </details>



              <div className="bottom-actions">

                <button
                  type="button"

                  className="secondary-button"

                  onClick={
                    downloadJsonReport
                  }
                >
                  ↓ Скачать JSON-отчёт
                </button>


                <button
                  type="button"

                  className="primary-button"

                  onClick={
                    resetAnalysis
                  }
                >
                  ← Анализировать другое видео
                </button>

              </div>

            </>
          )}

      </section>



      <footer className="footer">

        <div>
          SecureVision
        </div>

        <span>
          Local AI • No face recognition • Human review required
        </span>

      </footer>

    </main>
  );
}