"use client"

import { useState, useCallback, useEffect, useRef } from "react"
import { Header } from "@/components/header"
import { FileUpload } from "@/components/file-upload"
import { SchemaReview } from "@/components/schema-review"
import { ProcessingView } from "@/components/processing-view"
import { AugmentationPanel } from "@/components/augmentation-panel"
import type { SchemaReviewData, JobConfig, ColumnProfile } from "@/types/schema"
import { api, type AnalyzeResponse, type JobStatusResponse, type ImputationPreview } from "@/lib/api"
import { toast } from "@/hooks/use-toast"

export type AppState = "upload" | "schema" | "processing" | "complete" | "augment"

export interface ColumnInfo {
  name: string
  missingCount: number
  totalCount: number
  fillProgress: number
}

export interface JobInfo {
  id: string
  model: string
  status: "PROCESSING" | "COMPLETE" | "ERROR"
  progress: number
  eta: string
  stage: number
  columns: ColumnInfo[]
  logs: string[]
  previewData: (string | number | null)[][]
  downloadUrl?: string
  imputationPreview: ImputationPreview | null
}

// Convert API response to frontend schema format
function convertApiResponseToSchema(response: AnalyzeResponse): SchemaReviewData {
  return {
    jobId: response.job_id,
    filename: response.filename,
    sampleRows: response.sample_rows,
    totalNullRatio: response.total_null_ratio,
    columns: response.columns.map((col) => ({
      name: col.name,
      detectedType: col.detected_type,
      nullCount: col.null_count,
      nullRatio: col.null_ratio,
      uniqueCount: col.unique_count,
      example: col.example,
      recommendedAction: col.recommended_action,
      warnings: col.warnings,
      selectedType: col.detected_type,
      selectedRole: col.recommended_action === "IMPUTE" ? "TARGET" as const : "IGNORE" as const,
    })),
  }
}

// Map stage string to stage number
function getStageNumber(stage: string): number {
  const stages: Record<string, number> = {
    "Queued": 0,
    "Reading data": 0,
    "Preprocessing": 1,
    "Encoding": 1,
    "Imputing": 2,
    "Exporting": 3,
    "Complete": 4,
    "Failed": -1,
  }
  return stages[stage] ?? 0
}

export default function ImputeXPage() {
  const [appState, setAppState] = useState<AppState>("upload")
  const [schemaData, setSchemaData] = useState<SchemaReviewData | null>(null)
  const [jobInfo, setJobInfo] = useState<JobInfo | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const pollingRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pollingDelayRef = useRef(1000)
  const lastStatusRef = useRef<{
    progress: number
    stage: string
    status: string
    logsLen: number
  } | null>(null)

  const clearPolling = useCallback(() => {
    if (pollingRef.current) {
      clearTimeout(pollingRef.current)
      pollingRef.current = null
    }
  }, [])

  // Handle file upload -> analyze and go to schema review
  const handleFileAnalyze = useCallback(async (file: File) => {
    setIsLoading(true)
    setError(null)

    try {
      const response = await api.analyze(file)
      const schema = convertApiResponseToSchema(response)
      setSchemaData(schema)
      setAppState("schema")
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to analyze file")
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Handle back from schema review
  const handleBackToUpload = useCallback(() => {
    setAppState("upload")
    setSchemaData(null)
    setError(null)
  }, [])

  // Poll job status
  const pollJobStatus = useCallback(async (jobId: string) => {
    try {
      const status = await api.getJobStatus(jobId)

      // Calculate ETA
      const eta = status.progress >= 100
        ? "Complete"
        : `~${Math.max(1, Math.ceil((100 - status.progress) / 15))} min`

      setJobInfo((prev) => {
        if (!prev) return null
        return {
          ...prev,
          progress: status.progress,
          stage: getStageNumber(status.stage),
          logs: status.logs,
          eta,
          status: status.status === "COMPLETED" ? "COMPLETE"
            : status.status === "FAILED" ? "ERROR"
              : "PROCESSING",
          downloadUrl: status.download_url || undefined,
          imputationPreview: status.imputation_preview,
        }
      })

      if (status.status === "COMPLETED") {
        setAppState("complete")
        clearPolling()
        return
      } else if (status.status === "FAILED") {
        setError(status.error_message || "Job failed")
        clearPolling()
        return
      }

      const nextSnapshot = {
        progress: status.progress,
        stage: status.stage || "",
        status: status.status,
        logsLen: status.logs?.length ?? 0,
      }
      const prevSnapshot = lastStatusRef.current
      const changed = !prevSnapshot
        || prevSnapshot.progress !== nextSnapshot.progress
        || prevSnapshot.stage !== nextSnapshot.stage
        || prevSnapshot.status !== nextSnapshot.status
        || prevSnapshot.logsLen !== nextSnapshot.logsLen

      lastStatusRef.current = nextSnapshot
      const minDelay = 1000
      const maxDelay = 5000
      pollingDelayRef.current = changed
        ? minDelay
        : Math.min(maxDelay, pollingDelayRef.current * 2)

      clearPolling()
      pollingRef.current = setTimeout(() => {
        pollJobStatus(jobId)
      }, pollingDelayRef.current)
    } catch (err) {
      console.error("Failed to poll job status:", err)
      const maxDelay = 5000
      pollingDelayRef.current = Math.min(maxDelay, pollingDelayRef.current * 2)
      clearPolling()
      pollingRef.current = setTimeout(() => {
        pollJobStatus(jobId)
      }, pollingDelayRef.current)
    }
  }, [clearPolling])

  // Handle start job from schema review
  const handleStartJob = useCallback(async (config: JobConfig) => {
    if (!schemaData) return

    setIsLoading(true)
    setError(null)

    try {
      // Convert frontend config to API request format (model-specific params only)
      const { type } = config.model
      const p = config.model.params
      const hyperparameters: Record<string, unknown> =
        type === "KNN"
          ? { n_neighbors: p.nNeighbors, weights: p.weights, metric: p.metric }
          : type === "MEAN"
            ? {}
            : type === "NAOMI"
              ? {
                  hidden_dim: p.hiddenDim,
                  epochs: p.epochs,
                  lr: p.lr,
                  preview_updates: p.previewUpdates,
                  window_size: p.windowSize,
                  batch_size: p.batchSize,
                  n_layers: p.nLayers,
                  highest: p.highest,
                  clip: p.clip,
                }
              : type === "TOTEM"
                ? {
                    window_size: p.totemWindowSize,
                    normalization: p.totemNormalization,
                    merge_mode: p.totemMergeMode,
                    stride: p.totemStride,
                    preview_updates: p.totemPreviewUpdates,
                  }
                : { max_iter: p.maxIter, random_state: p.randomState, estimator: p.estimator }

      const request = {
        model_type: type as "MICE" | "KNN" | "MEAN" | "REGRESSION" | "NAOMI" | "TOTEM",
        hyperparameters,
        column_config: config.columns.map((col) => ({
          name: col.name,
          type: col.type,
          role: col.role,
        })),
      }

      await api.startJob(schemaData.jobId, request)

      // Initialize job info
      const mockColumns: ColumnInfo[] = config.columns
        .filter((col) => col.role === "TARGET")
        .map((col) => ({
          name: col.name,
          missingCount: Math.floor(Math.random() * 50) + 10,
          totalCount: 100,
          fillProgress: 0,
        }))

      const newJob: JobInfo = {
        id: schemaData.jobId,
        model: config.model.type,
        status: "PROCESSING",
        progress: 0,
        eta: "~3 min",
        stage: 0,
        columns: mockColumns,
        logs: [],
        previewData: [["Loading..."]],
        imputationPreview: null,
      }

      setJobInfo(newJob)
      setAppState("processing")

      // Start polling for job status (with backoff)
      pollingDelayRef.current = 1000
      lastStatusRef.current = null
      clearPolling()
      pollingRef.current = setTimeout(() => {
        pollJobStatus(schemaData.jobId)
      }, pollingDelayRef.current)

    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start job")
    } finally {
      setIsLoading(false)
    }
  }, [schemaData, pollJobStatus])

  // Cancel job
  const handleCancel = useCallback(async () => {
    clearPolling()

    if (jobInfo) {
      try {
        await api.cancelJob(jobInfo.id)
        toast({
          title: "작업이 중지됨",
          description: "보간 처리를 중단했습니다.",
        })
      } catch (err) {
        console.error("Failed to cancel job:", err)
        toast({
          title: "중지 실패",
          description: "작업 중지 요청에 실패했습니다.",
          variant: "destructive",
        })
      }
    }

    setAppState("upload")
    setJobInfo(null)
    setSchemaData(null)
    setError(null)
  }, [jobInfo])

  // New job
  const handleNewJob = useCallback(() => {
    clearPolling()

    setAppState("upload")
    setJobInfo(null)
    setSchemaData(null)
    setError(null)
  }, [])

  // Download result
  const handleDownload = useCallback(() => {
    if (jobInfo?.downloadUrl) {
      window.open(api.getDownloadUrl(jobInfo.id), "_blank")
    }
  }, [jobInfo])

  // Go to augmentation
  const handleAugment = useCallback(() => {
    setAppState("augment")
  }, [])

  // Back from augmentation
  const handleBackFromAugment = useCallback(() => {
    setAppState("complete")
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      clearPolling()
    }
  }, [clearPolling])

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="container mx-auto px-4 py-8 max-w-7xl">
        {error && (
          <div className="mb-6 p-4 bg-destructive/10 border border-destructive/30 rounded-xl text-destructive text-sm">
            {error}
            <button
              onClick={() => setError(null)}
              className="ml-4 underline hover:no-underline"
            >
              Dismiss
            </button>
          </div>
        )}

        {appState === "upload" && (
          <FileUpload onFileUpload={handleFileAnalyze} isLoading={isLoading} />
        )}

        {appState === "schema" && schemaData && (
          <SchemaReview
            data={schemaData}
            onBack={handleBackToUpload}
            onStartJob={handleStartJob}
            isLoading={isLoading}
          />
        )}

        {(appState === "processing" || appState === "complete") && jobInfo && (
          <ProcessingView
            jobInfo={jobInfo}
            onCancel={handleCancel}
            onNewJob={handleNewJob}
            onDownload={handleDownload}
            onAugment={handleAugment}
            isComplete={appState === "complete"}
          />
        )}

        {appState === "augment" && jobInfo && schemaData && (
          <AugmentationPanel
            jobId={jobInfo.id}
            columns={schemaData.columns}
            onBack={handleBackFromAugment}
          />
        )}
      </main>
    </div>
  )
}
