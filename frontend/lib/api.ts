const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1"

// Types matching backend schemas
export interface ColumnProfile {
  name: string
  detected_type: "ID" | "NUMERIC" | "CATEGORICAL" | "DATETIME"
  null_count: number
  null_ratio: number
  unique_count: number
  example: (string | number | null)[]
  recommended_action: "IMPUTE" | "IGNORE"
  warnings: string[]
}

export interface AnalyzeResponse {
  job_id: string
  filename: string
  sample_rows: number
  total_null_ratio: number
  columns: ColumnProfile[]
}

export interface StartJobRequest {
  model_type: "MICE" | "KNN" | "MEAN" | "REGRESSION" | "NAOMI" | "TOTEM"
  hyperparameters: Record<string, unknown>
  column_config: {
    name: string
    type: "ID" | "NUMERIC" | "CATEGORICAL" | "DATETIME"
    role: "TARGET" | "FEATURE" | "IGNORE"
  }[]
}

export interface ImputationPreviewData {
  column_name: string
  timestamps: string[]
  original: (number | null)[]
  imputed: (number | null)[]
}

export interface ImputationPreview {
  dates_with_missing: string[]
  preview_data: Record<string, ImputationPreviewData>
}

export interface JobStatusResponse {
  job_id: string
  status: "UPLOADED" | "REVIEWED" | "QUEUED" | "PROCESSING" | "COMPLETED" | "FAILED" | "CANCELED"
  progress: number
  stage: string
  download_url: string | null
  error_message: string | null
  logs: string[]
  imputation_preview: ImputationPreview | null
}

// Augmentation types
export interface AugmentRequest {
  label_column: string
  feature_columns: string[]
  categorical_feature_columns: string[]
  window_size: number
  stride: number
  k_neighbors: number
  sampling_strategy: string | number
  random_state: number
}

export interface AugmentColumnPreview {
  original: (number | null)[]
  synthetic: (number | null)[]
}

export interface AugmentPreview {
  original_count: number
  synthetic_count: number
  original_windows: number
  synthetic_windows: number
  columns: Record<string, AugmentColumnPreview>
  class_distribution: {
    before: Record<string, number>
    after: Record<string, number>
  }
}

export interface AugmentStatusResponse {
  job_id: string
  augment_status: "IDLE" | "QUEUED" | "PROCESSING" | "COMPLETED" | "FAILED"
  augment_progress: number
  augment_stage: string
  augment_download_url: string | null
  augment_error: string | null
  augment_preview: AugmentPreview | null
  logs: string[]
}

class ApiClient {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  async analyze(file: File): Promise<AnalyzeResponse> {
    const formData = new FormData()
    formData.append("file", file)

    const response = await fetch(`${this.baseUrl}/analyze`, {
      method: "POST",
      body: formData,
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Unknown error" }))
      throw new Error(error.detail || `HTTP ${response.status}`)
    }

    return response.json()
  }

  async startJob(jobId: string, request: StartJobRequest): Promise<{ job_id: string; status: string }> {
    const response = await fetch(`${this.baseUrl}/jobs/${jobId}/start`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Unknown error" }))
      throw new Error(error.detail || `HTTP ${response.status}`)
    }

    return response.json()
  }

  async getJobStatus(jobId: string): Promise<JobStatusResponse> {
    const response = await fetch(`${this.baseUrl}/jobs/${jobId}`)

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Unknown error" }))
      throw new Error(error.detail || `HTTP ${response.status}`)
    }

    return response.json()
  }

  async cancelJob(jobId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/jobs/${jobId}`, {
      method: "DELETE",
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Unknown error" }))
      throw new Error(error.detail || `HTTP ${response.status}`)
    }
  }

  getDownloadUrl(jobId: string): string {
    return `${this.baseUrl}/jobs/${jobId}/download`
  }

  // --- Augmentation API ---

  async startAugmentation(jobId: string, request: AugmentRequest): Promise<{ job_id: string; status: string }> {
    const response = await fetch(`${this.baseUrl}/jobs/${jobId}/augment`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Unknown error" }))
      throw new Error(error.detail || `HTTP ${response.status}`)
    }

    return response.json()
  }

  async getAugmentStatus(jobId: string): Promise<AugmentStatusResponse> {
    const response = await fetch(`${this.baseUrl}/jobs/${jobId}/augment`)

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Unknown error" }))
      throw new Error(error.detail || `HTTP ${response.status}`)
    }

    return response.json()
  }

  getAugmentDownloadUrl(jobId: string): string {
    return `${this.baseUrl}/jobs/${jobId}/augment/download`
  }
}

export const api = new ApiClient()
