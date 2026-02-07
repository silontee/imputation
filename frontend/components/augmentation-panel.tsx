"use client"

import { useState, useEffect, useCallback, useRef, useMemo } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts"
import {
  ArrowLeft,
  Play,
  Pause,
  Download,
  Loader2,
  CheckCircle2,
  Settings,
  BarChart3,
  Zap,
} from "lucide-react"
import type { ColumnProfile } from "@/types/schema"
import { api, type AugmentPreview, type AugmentStatusResponse } from "@/lib/api"

interface AugmentationPanelProps {
  jobId: string
  columns: ColumnProfile[]
  onBack: () => void
}

type AugmentPhase = "config" | "running" | "done"

export function AugmentationPanel({ jobId, columns, onBack }: AugmentationPanelProps) {
  // --- Config state ---
  const [labelColumn, setLabelColumn] = useState("")
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([])
  const [categoricalFeatures, setCategoricalFeatures] = useState<string[]>([])
  const [windowSize, setWindowSize] = useState(48)
  const [stride, setStride] = useState(4)
  const [kNeighbors, setKNeighbors] = useState(5)
  const [samplingStrategy, setSamplingStrategy] = useState("auto")
  const [randomState, setRandomState] = useState(42)

  // --- Run state ---
  const [phase, setPhase] = useState<AugmentPhase>("config")
  const [augStatus, setAugStatus] = useState<AugmentStatusResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const pollingRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pollingDelayRef = useRef(1000)
  const lastStatusRef = useRef<{
    status: string
    progress: number
    stage: string
    previewCols: number
  } | null>(null)

  // --- Graph state ---
  const [selectedColumn, setSelectedColumn] = useState("")
  const [showMode, setShowMode] = useState<"both" | "original" | "augmented">("both")
  const [isPlaying, setIsPlaying] = useState(false)
  const [animationIndex, setAnimationIndex] = useState(0)
  const animationRef = useRef<NodeJS.Timeout | null>(null)
  const [playSpeed, setPlaySpeed] = useState(1)

  const clearPolling = useCallback(() => {
    if (pollingRef.current) {
      clearTimeout(pollingRef.current)
      pollingRef.current = null
    }
  }, [])

  // Auto-select defaults
  useEffect(() => {
    const categoricals = columns.filter((c) => c.detectedType === "CATEGORICAL")
    const numerics = columns.filter(
      (c) => c.detectedType === "NUMERIC" && c.selectedRole !== "IGNORE"
    )

    // Default label: first categorical column
    if (!labelColumn && categoricals.length > 0) {
      setLabelColumn(categoricals[0].name)
    }

    // Default features: all numeric columns
    if (selectedFeatures.length === 0 && numerics.length > 0) {
      setSelectedFeatures(numerics.map((c) => c.name))
    }

    // Default categorical features
    if (categoricalFeatures.length === 0 && categoricals.length > 0) {
      setCategoricalFeatures(categoricals.map((c) => c.name))
    }
  }, [columns, labelColumn, selectedFeatures.length, categoricalFeatures.length])

  // Poll augmentation status
  const pollStatus = useCallback(async () => {
    try {
      const status = await api.getAugmentStatus(jobId)
      setAugStatus(status)

      if (status.augment_status === "COMPLETED") {
        setPhase("done")
        clearPolling()
        // Auto-select first preview column
        if (status.augment_preview?.columns) {
          const cols = Object.keys(status.augment_preview.columns)
          if (cols.length > 0 && !selectedColumn) {
            setSelectedColumn(cols[0])
          }
        }
        return
      } else if (status.augment_status === "FAILED") {
        setError(status.augment_error || "Augmentation failed")
        setPhase("config")
        clearPolling()
        return
      }

      const previewCols = status.augment_preview?.columns
        ? Object.keys(status.augment_preview.columns).length
        : 0
      const nextSnapshot = {
        status: status.augment_status,
        progress: status.augment_progress,
        stage: status.augment_stage || "",
        previewCols,
      }
      const prevSnapshot = lastStatusRef.current
      const changed = !prevSnapshot
        || prevSnapshot.status !== nextSnapshot.status
        || prevSnapshot.progress !== nextSnapshot.progress
        || prevSnapshot.stage !== nextSnapshot.stage
        || prevSnapshot.previewCols !== nextSnapshot.previewCols

      lastStatusRef.current = nextSnapshot
      const minDelay = 1000
      const maxDelay = 5000
      pollingDelayRef.current = changed
        ? minDelay
        : Math.min(maxDelay, pollingDelayRef.current * 2)

      clearPolling()
      pollingRef.current = setTimeout(pollStatus, pollingDelayRef.current)
    } catch (err) {
      console.error("Failed to poll augment status:", err)
      const maxDelay = 5000
      pollingDelayRef.current = Math.min(maxDelay, pollingDelayRef.current * 2)
      clearPolling()
      pollingRef.current = setTimeout(pollStatus, pollingDelayRef.current)
    }
  }, [jobId, selectedColumn, clearPolling])

  // Start augmentation
  const handleStart = useCallback(async () => {
    if (!labelColumn || selectedFeatures.length === 0) {
      setError("Label column and feature columns are required")
      return
    }

    setIsSubmitting(true)
    setError(null)

    try {
      await api.startAugmentation(jobId, {
        label_column: labelColumn,
        feature_columns: selectedFeatures,
        categorical_feature_columns: categoricalFeatures,
        window_size: windowSize,
        stride,
        k_neighbors: kNeighbors,
        sampling_strategy: samplingStrategy,
        random_state: randomState,
      })

      setPhase("running")
      pollingDelayRef.current = 1000
      lastStatusRef.current = null
      clearPolling()
      pollingRef.current = setTimeout(pollStatus, pollingDelayRef.current)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start augmentation")
    } finally {
      setIsSubmitting(false)
    }
  }, [
    jobId, labelColumn, selectedFeatures, categoricalFeatures,
    windowSize, stride, kNeighbors, samplingStrategy, randomState, pollStatus,
  ])

  // Download
  const handleDownload = useCallback(() => {
    window.open(api.getAugmentDownloadUrl(jobId), "_blank")
  }, [jobId])

  // Toggle feature
  const toggleFeature = (colName: string) => {
    setSelectedFeatures((prev) =>
      prev.includes(colName) ? prev.filter((c) => c !== colName) : [...prev, colName]
    )
  }

  // Toggle categorical
  const toggleCategorical = (colName: string) => {
    setCategoricalFeatures((prev) =>
      prev.includes(colName) ? prev.filter((c) => c !== colName) : [...prev, colName]
    )
  }

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      clearPolling()
      if (animationRef.current) clearInterval(animationRef.current)
    }
  }, [clearPolling])

  // Chart data
  const chartData = useMemo(() => {
    const preview = augStatus?.augment_preview
    if (!preview?.columns || !selectedColumn || !preview.columns[selectedColumn]) {
      return []
    }

    const col = preview.columns[selectedColumn]
    const maxLen = Math.max(col.original.length, col.synthetic.length)
    const limit = isPlaying ? animationIndex : maxLen

    const data = []
    for (let i = 0; i < Math.min(limit, maxLen); i++) {
      data.push({
        index: i,
        original: i < col.original.length ? col.original[i] : null,
        synthetic: i < col.synthetic.length ? col.synthetic[i] : null,
      })
    }
    return data
  }, [augStatus, selectedColumn, isPlaying, animationIndex])

  // Animation
  useEffect(() => {
    if (isPlaying && chartData.length > 0) {
      const totalPoints = augStatus?.augment_preview?.columns[selectedColumn]
        ? Math.max(
            augStatus.augment_preview.columns[selectedColumn].original.length,
            augStatus.augment_preview.columns[selectedColumn].synthetic.length
          )
        : 0

      animationRef.current = setInterval(() => {
        setAnimationIndex((prev) => {
          if (prev >= totalPoints) {
            setIsPlaying(false)
            return prev
          }
          return prev + 2
        })
      }, 50 / playSpeed)

      return () => {
        if (animationRef.current) clearInterval(animationRef.current)
      }
    }
  }, [isPlaying, playSpeed, augStatus, selectedColumn, chartData.length])

  const handlePlay = () => {
    if (!isPlaying) {
      setAnimationIndex(0)
    }
    setIsPlaying(!isPlaying)
  }

  // Columns available
  const allColumnNames = columns.map((c) => c.name)
  const previewColumns = augStatus?.augment_preview?.columns
    ? Object.keys(augStatus.augment_preview.columns)
    : []

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="sm" onClick={onBack}>
            <ArrowLeft className="w-4 h-4 mr-1" />
            Back
          </Button>
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-primary" />
            <h2 className="text-xl font-semibold">Data Augmentation (SMOTENC)</h2>
          </div>
        </div>
        {phase === "done" && (
          <Button onClick={handleDownload} className="gap-2">
            <Download className="w-4 h-4" />
            Download Augmented CSV
          </Button>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="p-4 bg-destructive/10 border border-destructive/30 rounded-xl text-destructive text-sm">
          {error}
          <button onClick={() => setError(null)} className="ml-4 underline hover:no-underline">
            Dismiss
          </button>
        </div>
      )}

      {/* Progress (while running) */}
      {phase === "running" && augStatus && (
        <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
          <CardContent className="p-6">
            <div className="flex items-center gap-4 mb-4">
              <Loader2 className="w-5 h-5 text-primary animate-spin" />
              <span className="text-sm font-medium">
                {augStatus.augment_stage || "Processing..."}
              </span>
            </div>
            <Progress value={augStatus.augment_progress} className="h-2" />
            <div className="flex justify-between mt-2 text-xs text-muted-foreground">
              <span>Progress</span>
              <span>{augStatus.augment_progress}%</span>
            </div>
            {/* Logs */}
            {augStatus.logs.length > 0 && (
              <div className="mt-4 h-32 overflow-y-auto rounded-lg bg-background/50 border border-border p-3 font-mono text-xs space-y-1">
                {augStatus.logs
                  .filter((l) => l.includes("[Augment]"))
                  .map((log, i) => (
                    <div key={i} className="text-muted-foreground">{log}</div>
                  ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Config + Results Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Settings Panel */}
        <Card className="border-border/50 bg-card/50 backdrop-blur-sm lg:col-span-1">
          <CardHeader className="pb-4">
            <CardTitle className="text-base font-medium flex items-center gap-2">
              <Settings className="w-4 h-4 text-primary" />
              Augmentation Settings
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Label Column */}
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">Label Column (required)</Label>
              <Select value={labelColumn} onValueChange={setLabelColumn} disabled={phase !== "config"}>
                <SelectTrigger className="h-9 text-sm">
                  <SelectValue placeholder="Select label column" />
                </SelectTrigger>
                <SelectContent>
                  {allColumnNames.map((col) => (
                    <SelectItem key={col} value={col}>{col}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Feature Columns */}
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">
                Feature Columns ({selectedFeatures.length} selected)
              </Label>
              <div className="max-h-32 overflow-y-auto border border-border rounded-lg p-2 space-y-1">
                {allColumnNames
                  .filter((c) => c !== labelColumn)
                  .map((col) => (
                    <label key={col} className="flex items-center gap-2 text-xs cursor-pointer hover:bg-secondary/50 rounded px-1 py-0.5">
                      <input
                        type="checkbox"
                        checked={selectedFeatures.includes(col)}
                        onChange={() => toggleFeature(col)}
                        disabled={phase !== "config"}
                        className="rounded"
                      />
                      <span>{col}</span>
                    </label>
                  ))}
              </div>
            </div>

            {/* Categorical Features */}
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">
                Categorical Features ({categoricalFeatures.length})
              </Label>
              <div className="max-h-24 overflow-y-auto border border-border rounded-lg p-2 space-y-1">
                {allColumnNames
                  .filter((c) => c !== labelColumn)
                  .map((col) => (
                    <label key={col} className="flex items-center gap-2 text-xs cursor-pointer hover:bg-secondary/50 rounded px-1 py-0.5">
                      <input
                        type="checkbox"
                        checked={categoricalFeatures.includes(col)}
                        onChange={() => toggleCategorical(col)}
                        disabled={phase !== "config"}
                        className="rounded"
                      />
                      <span>{col}</span>
                    </label>
                  ))}
              </div>
            </div>

            {/* Parameters */}
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">Window Size</Label>
                <Input
                  type="number"
                  value={windowSize}
                  onChange={(e) => setWindowSize(Number(e.target.value))}
                  min={4}
                  max={1024}
                  className="h-9 text-sm"
                  disabled={phase !== "config"}
                />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">Stride</Label>
                <Input
                  type="number"
                  value={stride}
                  onChange={(e) => setStride(Number(e.target.value))}
                  min={1}
                  max={512}
                  className="h-9 text-sm"
                  disabled={phase !== "config"}
                />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">k Neighbors</Label>
                <Input
                  type="number"
                  value={kNeighbors}
                  onChange={(e) => setKNeighbors(Number(e.target.value))}
                  min={1}
                  max={50}
                  className="h-9 text-sm"
                  disabled={phase !== "config"}
                />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">Random State</Label>
                <Input
                  type="number"
                  value={randomState}
                  onChange={(e) => setRandomState(Number(e.target.value))}
                  min={0}
                  className="h-9 text-sm"
                  disabled={phase !== "config"}
                />
              </div>
            </div>

            {/* Sampling Strategy */}
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">Sampling Strategy</Label>
              <Select value={samplingStrategy} onValueChange={setSamplingStrategy} disabled={phase !== "config"}>
                <SelectTrigger className="h-9 text-sm">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">auto</SelectItem>
                  <SelectItem value="minority">minority</SelectItem>
                  <SelectItem value="not majority">not majority</SelectItem>
                  <SelectItem value="not minority">not minority</SelectItem>
                  <SelectItem value="all">all</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Start Button */}
            {phase === "config" && (
              <Button
                onClick={handleStart}
                disabled={isSubmitting || !labelColumn || selectedFeatures.length === 0}
                className="w-full gap-2"
              >
                {isSubmitting ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Zap className="w-4 h-4" />
                )}
                Start Augmentation
              </Button>
            )}
          </CardContent>
        </Card>

        {/* Graph + Results */}
        <div className="lg:col-span-2 space-y-6">
          {/* Graph */}
          {phase === "done" && augStatus?.augment_preview && (
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base font-medium flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-primary" />
                    Original vs Augmented
                  </CardTitle>
                  <div className="flex items-center gap-2">
                    {previewColumns.length > 0 && (
                      <Select value={selectedColumn} onValueChange={setSelectedColumn}>
                        <SelectTrigger className="w-40 h-8 text-xs">
                          <SelectValue placeholder="Column" />
                        </SelectTrigger>
                        <SelectContent>
                          {previewColumns.map((col) => (
                            <SelectItem key={col} value={col}>{col}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    )}
                    <Select
                      value={showMode}
                      onValueChange={(v) => setShowMode(v as "both" | "original" | "augmented")}
                    >
                      <SelectTrigger className="w-28 h-8 text-xs">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="both">Both</SelectItem>
                        <SelectItem value="original">Original</SelectItem>
                        <SelectItem value="augmented">Augmented</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="pt-0">
                {/* Legend */}
                <div className="flex items-center gap-6 mb-3 text-xs">
                  <div className="flex items-center gap-1.5">
                    <div className="w-5 h-0.5 bg-primary rounded" />
                    <span className="text-muted-foreground">Original</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="w-5 h-0.5 rounded" style={{ backgroundColor: "hsl(var(--chart-2))" }} />
                    <span className="text-muted-foreground">Augmented (Synthetic)</span>
                  </div>
                </div>

                {/* Chart */}
                <div className="h-[300px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="hsl(var(--border))"
                        strokeOpacity={0.5}
                      />
                      <XAxis
                        dataKey="index"
                        tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }}
                        axisLine={{ stroke: "hsl(var(--border))" }}
                        tickLine={{ stroke: "hsl(var(--border))" }}
                      />
                      <YAxis
                        tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }}
                        axisLine={{ stroke: "hsl(var(--border))" }}
                        tickLine={{ stroke: "hsl(var(--border))" }}
                        width={60}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: "8px",
                          boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
                        }}
                        labelStyle={{ color: "hsl(var(--foreground))" }}
                      />
                      {(showMode === "both" || showMode === "original") && (
                        <Line
                          type="monotone"
                          dataKey="original"
                          stroke="hsl(var(--primary))"
                          strokeWidth={2}
                          dot={false}
                          connectNulls={false}
                          name="Original"
                        />
                      )}
                      {(showMode === "both" || showMode === "augmented") && (
                        <Line
                          type="monotone"
                          dataKey="synthetic"
                          stroke="hsl(var(--chart-2))"
                          strokeWidth={2}
                          strokeDasharray="5 3"
                          dot={false}
                          connectNulls={false}
                          name="Augmented"
                        />
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Playback Controls */}
                <div className="flex items-center gap-3 mt-3">
                  <Button variant="outline" size="sm" onClick={handlePlay} className="gap-1">
                    {isPlaying ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
                    {isPlaying ? "Pause" : "Play"}
                  </Button>
                  <Select value={String(playSpeed)} onValueChange={(v) => setPlaySpeed(Number(v))}>
                    <SelectTrigger className="w-20 h-8 text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="0.5">x0.5</SelectItem>
                      <SelectItem value="1">x1</SelectItem>
                      <SelectItem value="2">x2</SelectItem>
                      <SelectItem value="4">x4</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Stats */}
          {phase === "done" && augStatus?.augment_preview && (
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader className="pb-4">
                <CardTitle className="text-base font-medium">Augmentation Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-foreground">
                      {augStatus.augment_preview.original_count?.toLocaleString()}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">Original Rows</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-primary">
                      {augStatus.augment_preview.synthetic_count?.toLocaleString()}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">Synthetic Rows</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-foreground">
                      {augStatus.augment_preview.original_windows}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">Original Windows</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-primary">
                      {augStatus.augment_preview.synthetic_windows}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">Synthetic Windows</div>
                  </div>
                </div>

                {/* Class Distribution */}
                {augStatus.augment_preview.class_distribution && (
                  <div className="mt-6 space-y-3">
                    <h4 className="text-sm font-medium text-muted-foreground">Class Distribution</h4>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-xs text-muted-foreground mb-2">Before</p>
                        <div className="space-y-1">
                          {Object.entries(augStatus.augment_preview.class_distribution.before || {}).map(([cls, count]) => (
                            <div key={cls} className="flex justify-between text-xs">
                              <Badge variant="outline" className="font-mono">{cls}</Badge>
                              <span className="text-foreground">{count}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground mb-2">After</p>
                        <div className="space-y-1">
                          {Object.entries(augStatus.augment_preview.class_distribution.after || {}).map(([cls, count]) => (
                            <div key={cls} className="flex justify-between text-xs">
                              <Badge variant="outline" className="font-mono">{cls}</Badge>
                              <span className="text-primary font-medium">{count}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Empty state when in config */}
          {phase === "config" && (
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardContent className="p-12 flex flex-col items-center justify-center text-center">
                <BarChart3 className="w-12 h-12 text-muted-foreground/30 mb-4" />
                <h3 className="text-lg font-medium text-muted-foreground">
                  Configure & Start Augmentation
                </h3>
                <p className="text-sm text-muted-foreground/70 mt-2 max-w-md">
                  Select a label column, feature columns, and configure SMOTENC parameters.
                  The augmented data will be visualized here after processing.
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
