export type AnalysisStatus =
  | "processing"
  | "done"
  | "failed";


export interface SubmitAnalysisResponse {
  job_id: string;
  analysis_type: string;
  status: AnalysisStatus;
  file_name: string;
}


export interface CashDisciplineSummary {
  verdict: string | null;
  violation_detected: boolean;
  violation_count: number;
}


export interface WorkplaceSummary {
  presence_percent?: number | null;
  present_sec?: number | null;
  away_sec?: number | null;
  unknown_sec?: number | null;

  sitting_sec?: number | null;
  standing_sec?: number | null;
  unknown_posture_sec?: number | null;

  sitting_percent_of_known_posture?:
    number | null;

  standing_percent_of_known_posture?:
    number | null;

  absence_count?: number | null;
  longest_absence_sec?: number | null;
  samples?: number | null;
  tracks_created?: number | null;
}


export type EvidenceFrame =
  | string
  | {
      base64?: string;
      image_base64?: string;
      jpeg_base64?: string;
      frame_base64?: string;
      timecode_sec?: number;
      timestamp_sec?: number;
      [key: string]: unknown;
    };


export interface Violation {
  type?: string;
  timecode_sec?: number;
  confidence?: number;

  description?: string;

  legal_status?: string;

  automated_legal_conclusion?:
    boolean;

  evidence?: EvidenceFrame[];

  evidence_frames?: EvidenceFrame[];

  legal_context?: unknown;

  [key: string]: unknown;
}


export interface CashDisciplineResult {
  verdict?: string;

  violation_detected?: boolean;

  violations?: Violation[];

  legal_context?: unknown;

  analysis?: unknown;

  processing_time_sec?: number;

  [key: string]: unknown;
}


export interface WorkplaceTimelineItem {
  timecode_sec: number;

  sample_duration_sec:
    number;

  state:
    | "present"
    | "away"
    | "unknown"
    | string;

  posture:
    | "sitting"
    | "standing"
    | "unknown"
    | "away"
    | string;

  cashier_present?: boolean;

  active_cashier_track_id?:
    number | null;

  detector_confidence?:
    number | null;

  posture_confidence?:
    number | null;

  [key: string]: unknown;
   frame_valid?: boolean;
  people?: WorkplaceTrackedPerson[];
}


export interface AbsenceInterval {
  start_sec: number;
  end_sec: number;
  duration_sec: number;
}


export interface WorkplaceResult {

   video?: WorkplaceVideoMetadata;
  configuration?: WorkplaceConfiguration;
  visualization?: WorkplaceVisualization;
  summary?: WorkplaceSummary;

  timeline?:
    WorkplaceTimelineItem[];

  absence_intervals?:
    AbsenceInterval[];

  processing_time_sec?: number;

  methodology?: unknown;

  tracking?: unknown;

  [key: string]: unknown;
}


export interface FullAnalysisSummary {
  cash_discipline:
    CashDisciplineSummary;

  workplace:
    WorkplaceSummary;
}


export interface ProcessingSummary {
  cash_discipline_sec:
    number;

  workplace_sec:
    number;

  total_sec:
    number;
}


export interface FullAnalysisResult {
  status: string;

  analysis_type: string;

  file_name: string;

  summary:
    FullAnalysisSummary;

  cash_discipline:
    CashDisciplineResult;

  workplace:
    WorkplaceResult;

  processing:
    ProcessingSummary;
}


export interface FullAnalysisJob {
  job_id: string;
  analysis_type: string;
  status: AnalysisStatus;
  file_name: string;
  created_at?: string;
  updated_at?: string;
  result:
    FullAnalysisResult | null;
  error:
    string | null;
}
export interface WorkplaceTrackedPerson {
  track_id: number;
  confidence: number;
  in_cashier_zone: boolean;
  bbox: [number, number, number, number];
  center?: [number, number];
}

export interface WorkplaceVideoMetadata {
  duration_sec: number;
  fps: number;
  frame_count: number;
  width: number;
  height: number;
}

export interface WorkplaceCashierZone {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface WorkplaceConfiguration {
  sample_interval_sec?: number;
  confidence_threshold?: number;
  min_absence_sec?: number;
  cashier_zone?: WorkplaceCashierZone;
  device?: string;
}

export interface WorkplaceVisualization {
  bbox_format: string;
  bbox_coordinate_system: string;
  cashier_zone_coordinate_system: string;
  sample_interval_sec: number;
  anonymous_tracking: boolean;
  face_recognition: boolean;
  intermediate_frames_are_estimates: boolean;
}