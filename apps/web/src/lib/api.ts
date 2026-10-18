import type {
  FullAnalysisJob,
  SubmitAnalysisResponse,
} from "./types";


const API_BASE_URL =
  process.env
    .NEXT_PUBLIC_API_URL ??
  "http://127.0.0.1:4000";


async function parseResponse<T>(
  response: Response,
): Promise<T> {
  const text =
    await response.text();

  let data: unknown = {};

  try {
    data = text
      ? JSON.parse(text)
      : {};
  } catch {
    data = {
      message: text,
    };
  }

  if (!response.ok) {
    let message =
      `Request failed with status ${response.status}`;

    if (
      typeof data ===
        "object" &&
      data !== null &&
      "message" in data
    ) {
      const apiMessage =
        (
          data as {
            message?: unknown;
          }
        ).message;

      if (
        typeof apiMessage ===
        "string"
      ) {
        message =
          apiMessage;
      }
    }

    throw new Error(
      message,
    );
  }

  return data as T;
}


export async function submitFullAnalysis(
  file: File,
): Promise<SubmitAnalysisResponse> {
  const formData =
    new FormData();

  formData.append(
    "file",
    file,
  );

  const response =
    await fetch(
      `${API_BASE_URL}/api/analysis/full`,
      {
        method: "POST",
        body: formData,
      },
    );

  return parseResponse<
    SubmitAnalysisResponse
  >(response);
}


export async function getFullAnalysisJob(
  jobId: string,
): Promise<FullAnalysisJob> {
  const response =
    await fetch(
      `${API_BASE_URL}/api/analysis/full/${encodeURIComponent(
        jobId,
      )}`,
      {
        method: "GET",
        cache: "no-store",
      },
    );

  return parseResponse<
    FullAnalysisJob
  >(response);
}


interface PollOptions {
  intervalMs?: number;
  timeoutMs?: number;

  onUpdate?: (
    job: FullAnalysisJob,
  ) => void;
}


export async function waitForFullAnalysis(
  jobId: string,
  options: PollOptions = {},
): Promise<FullAnalysisJob> {
  const {
    intervalMs = 1000,
    timeoutMs = 180_000,
    onUpdate,
  } = options;

  const startedAt =
    Date.now();

  while (true) {
    const job =
      await getFullAnalysisJob(
        jobId,
      );

    onUpdate?.(
      job,
    );

    if (
      job.status ===
      "done"
    ) {
      return job;
    }

    if (
      job.status ===
      "failed"
    ) {
      throw new Error(
        job.error ??
          "Video analysis failed.",
      );
    }

    if (
      Date.now() -
        startedAt >
      timeoutMs
    ) {
      throw new Error(
        "Analysis timeout. Please try again.",
      );
    }

    await new Promise<void>(
      (resolve) => {
        window.setTimeout(
          resolve,
          intervalMs,
        );
      },
    );
  }
}