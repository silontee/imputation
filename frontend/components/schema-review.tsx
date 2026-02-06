"use client"

import { useEffect, useState, useMemo } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  ArrowLeft,
  Play,
  Search,
  AlertTriangle,
  FileSpreadsheet,
  Settings2,
  Info,
} from "lucide-react"
import type {
  SchemaReviewData,
  ColumnProfile,
  ColumnType,
  ColumnRole,
  JobConfig,
  ModelConfig,
} from "@/types/schema"

interface SchemaReviewProps {
  data: SchemaReviewData
  onBack: () => void
  onStartJob: (config: JobConfig) => void
  isLoading?: boolean
}

const typeOptions: { value: ColumnType; label: string }[] = [
  { value: "ID", label: "ID" },
  { value: "NUMERIC", label: "NUM" },
  { value: "CATEGORICAL", label: "CAT" },
  { value: "DATETIME", label: "DATE" },
]

const typeBadgeVariant: Record<ColumnType, "id" | "numeric" | "categorical" | "datetime"> = {
  ID: "id",
  NUMERIC: "numeric",
  CATEGORICAL: "categorical",
  DATETIME: "datetime",
}

export function SchemaReview({ data, onBack, onStartJob, isLoading = false }: SchemaReviewProps) {
  const [columns, setColumns] = useState<ColumnProfile[]>(data.columns)
  const [searchQuery, setSearchQuery] = useState("")
  const [filterMode, setFilterMode] = useState<"all" | "missing">("all")
  const [model, setModel] = useState<ModelConfig["type"]>("MICE")
  // MICE / REGRESSION params
  const [maxIter, setMaxIter] = useState(10)
  const [randomState, setRandomState] = useState(42)
  const [estimator, setEstimator] = useState<"bayesian_ridge" | "random_forest" | "extra_trees">("bayesian_ridge")
  // KNN params
  const [nNeighbors, setNNeighbors] = useState(5)
  const [weights, setWeights] = useState<"uniform" | "distance">("uniform")
  const [metric, setMetric] = useState<"nan_euclidean" | "euclidean">("nan_euclidean")
  // NAOMI params
  const [hiddenDim, setHiddenDim] = useState(64)
  const [naomiEpochs, setNaomiEpochs] = useState(50)
  const [naomiLr, setNaomiLr] = useState(0.001)
  const [naomiPreviewUpdates, setNaomiPreviewUpdates] = useState(10)
  const [naomiWindowSize, setNaomiWindowSize] = useState(50)
  const [naomiBatchSize, setNaomiBatchSize] = useState(64)
  const [naomiLayers, setNaomiLayers] = useState(2)
  const [naomiHighest, setNaomiHighest] = useState(8)
  const [naomiClip, setNaomiClip] = useState(10)
  // TOTEM params
  const [totemWindowSize, setTotemWindowSize] = useState(96)
  const [totemNormalization, setTotemNormalization] = useState<"zscore" | "minmax">("zscore")
  const [totemMergeMode, setTotemMergeMode] = useState<"non_overlap" | "overlap">("non_overlap")
  const [totemStride, setTotemStride] = useState(48)
  const [totemPreviewUpdates, setTotemPreviewUpdates] = useState(10)
  // Advanced options toggle
  const [showAdvanced, setShowAdvanced] = useState(false)

  const totemStrideOptions = useMemo(() => {
    const candidates = [
      Math.max(1, Math.floor(totemWindowSize / 2)),
      Math.max(1, Math.floor(totemWindowSize / 4)),
      Math.max(1, Math.floor(totemWindowSize / 8)),
    ].filter((v) => v < totemWindowSize)
    return [...new Set(candidates)]
  }, [totemWindowSize])

  useEffect(() => {
    if (totemMergeMode !== "overlap") return
    if (totemStride <= 0 || totemStride >= totemWindowSize || !totemStrideOptions.includes(totemStride)) {
      setTotemStride(totemStrideOptions[0] ?? Math.max(1, Math.floor(totemWindowSize / 2)))
    }
  }, [totemMergeMode, totemStride, totemStrideOptions, totemWindowSize])

  const filteredColumns = useMemo(() => {
    return columns.filter((col) => {
      const matchesSearch = col.name.toLowerCase().includes(searchQuery.toLowerCase())
      const matchesFilter = filterMode === "all" || col.nullRatio > 0
      return matchesSearch && matchesFilter
    })
  }, [columns, searchQuery, filterMode])

  const warnings = useMemo(() => {
    return columns.filter((col) => col.warnings.length > 0)
  }, [columns])

  const targetCount = useMemo(() => columns.filter((c) => c.selectedRole === "TARGET").length, [columns])
  const featureCount = useMemo(() => columns.filter((c) => c.selectedRole === "FEATURE").length, [columns])
  const ignoreCount = useMemo(() => columns.filter((c) => c.selectedRole === "IGNORE").length, [columns])

  const handleTypeChange = (columnName: string, newType: ColumnType) => {
    setColumns((prev) =>
      prev.map((col) =>
        col.name === columnName ? { ...col, selectedType: newType } : col
      )
    )
  }

  const nextRole: Record<ColumnRole, ColumnRole> = { TARGET: "FEATURE", FEATURE: "IGNORE", IGNORE: "TARGET" }

  const handleRoleChange = (columnName: string) => {
    setColumns((prev) =>
      prev.map((col) =>
        col.name === columnName ? { ...col, selectedRole: nextRole[col.selectedRole] } : col
      )
    )
  }

  const handleBulkRole = (role: ColumnRole, filter?: (col: ColumnProfile) => boolean) => {
    setColumns((prev) =>
      prev.map((col) => (filter ? (filter(col) ? { ...col, selectedRole: role } : col) : { ...col, selectedRole: role }))
    )
  }

  const handleStartJob = () => {
    const params: ModelConfig["params"] =
      model === "KNN"
        ? { nNeighbors, weights, metric }
          : model === "MEAN"
            ? {}
            : model === "NAOMI"
              ? {
                  hiddenDim,
                  epochs: naomiEpochs,
                  lr: naomiLr,
                  previewUpdates: naomiPreviewUpdates,
                  windowSize: naomiWindowSize,
                  batchSize: naomiBatchSize,
                  nLayers: naomiLayers,
                  highest: naomiHighest,
                  clip: naomiClip,
                }
              : model === "TOTEM"
                ? {
                    totemWindowSize,
                    totemNormalization,
                    totemMergeMode,
                    totemStride: totemMergeMode === "overlap" ? totemStride : totemWindowSize,
                    totemPreviewUpdates,
                  }
                : { maxIter, randomState, estimator }

    const config: JobConfig = {
      model: { type: model, params },
      columns: columns.map((col) => ({
        name: col.name,
        type: col.selectedType,
        role: col.selectedRole,
      })),
    }
    onStartJob(config)
  }

  return (
    <div className="space-y-6">
      {/* Header Info */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary/10 border border-primary/30 flex items-center justify-center">
            <FileSpreadsheet className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h2 className="text-xl font-medium text-foreground">Schema Review</h2>
            <div className="flex items-center gap-4 text-sm text-muted-foreground">
              <span>{data.filename}</span>
              <span className="text-border">|</span>
              <span>{data.sampleRows.toLocaleString()} rows (sample)</span>
              <span className="text-border">|</span>
              <span>{(data.totalNullRatio * 100).toFixed(1)}% missing</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Column Table */}
        <div className="lg:col-span-2 space-y-4">
          <Card className="border-border/50 bg-card/50">
            <CardHeader className="pb-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <CardTitle className="text-base font-medium">Columns ({columns.length})</CardTitle>
                <div className="flex items-center gap-2">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <Input
                      placeholder="Search columns..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-9 h-9 w-48 bg-secondary/50 border-border/50"
                    />
                  </div>
                  <Select value={filterMode} onValueChange={(v) => setFilterMode(v as "all" | "missing")}>
                    <SelectTrigger className="w-32 h-9 bg-secondary/50 border-border/50">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All columns</SelectItem>
                      <SelectItem value="missing">Missing &gt; 0</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="rounded-lg border border-border overflow-hidden">
                <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
                  <Table>
                    <TableHeader className="sticky top-0 bg-card z-10">
                      <TableRow className="hover:bg-transparent">
                        <TableHead className="w-[180px]">Name</TableHead>
                        <TableHead className="w-[100px]">Type</TableHead>
                        <TableHead className="w-[100px]">Role</TableHead>
                        <TableHead className="w-[80px] text-right">Missing</TableHead>
                        <TableHead className="w-[80px] text-right">Unique</TableHead>
                        <TableHead>Example</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredColumns.map((col) => (
                        <TableRow key={col.name} className="group">
                          <TableCell className="font-mono text-sm">
                            <div className="flex items-center gap-2">
                              {col.warnings.length > 0 && (
                                <AlertTriangle className="w-3.5 h-3.5 text-warning shrink-0" />
                              )}
                              <span className="truncate">{col.name}</span>
                            </div>
                          </TableCell>
                          <TableCell>
                            <Select
                              value={col.selectedType}
                              onValueChange={(v) => handleTypeChange(col.name, v as ColumnType)}
                            >
                              <SelectTrigger className="h-7 w-20 text-xs bg-transparent border-transparent hover:bg-secondary/50 hover:border-border/50">
                                <Badge variant={typeBadgeVariant[col.selectedType]} className="text-[10px] px-1.5">
                                  {typeOptions.find((t) => t.value === col.selectedType)?.label}
                                </Badge>
                              </SelectTrigger>
                              <SelectContent>
                                {typeOptions.map((opt) => (
                                  <SelectItem key={opt.value} value={opt.value}>
                                    {opt.label}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          </TableCell>
                          <TableCell>
                            <button
                              onClick={() => handleRoleChange(col.name)}
                              className={`px-2 py-1 rounded-md text-xs font-medium transition-colors ${
                                col.selectedRole === "TARGET"
                                  ? "bg-primary/15 text-primary hover:bg-primary/25"
                                  : col.selectedRole === "FEATURE"
                                    ? "bg-blue-500/15 text-blue-500 hover:bg-blue-500/25"
                                    : "bg-secondary text-muted-foreground hover:bg-secondary/80"
                              }`}
                            >
                              {col.selectedRole}
                            </button>
                          </TableCell>
                          <TableCell className="text-right">
                            <span className={col.nullRatio > 0.1 ? "text-warning" : "text-muted-foreground"}>
                              {(col.nullRatio * 100).toFixed(1)}%
                            </span>
                          </TableCell>
                          <TableCell className="text-right text-muted-foreground">
                            {col.uniqueCount.toLocaleString()}
                          </TableCell>
                          <TableCell className="text-muted-foreground text-xs font-mono truncate max-w-[150px]">
                            {col.example.slice(0, 3).join(", ")}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Warnings */}
          {warnings.length > 0 && (
            <Card className="border-warning/30 bg-warning/5">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2 text-warning">
                  <AlertTriangle className="w-4 h-4" />
                  Warnings ({warnings.length})
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <ul className="space-y-1">
                  {warnings.slice(0, 5).map((col) => (
                    <li key={col.name} className="text-sm text-muted-foreground">
                      <span className="font-mono text-foreground">{col.name}</span>: {col.warnings[0]}
                    </li>
                  ))}
                  {warnings.length > 5 && (
                    <li className="text-sm text-muted-foreground">
                      ... and {warnings.length - 5} more warnings
                    </li>
                  )}
                </ul>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right: Settings Panel */}
        <div className="space-y-4">
          <Card className="border-border/50 bg-card/50">
            <CardHeader className="pb-4">
              <CardTitle className="text-base font-medium flex items-center gap-2">
                <Settings2 className="w-4 h-4 text-primary" />
                Model Settings
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm text-foreground">Imputation Model</label>
                <Select value={model} onValueChange={(v) => { setModel(v as ModelConfig["type"]); setShowAdvanced(false) }}>
                  <SelectTrigger className="bg-secondary/50 border-border/50">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="MICE">MICE (Multiple Imputation)</SelectItem>
                    <SelectItem value="KNN">KNN Imputer</SelectItem>
                    <SelectItem value="MEAN">Mean/Mode</SelectItem>
                    <SelectItem value="REGRESSION">Regression</SelectItem>
                    <SelectItem value="NAOMI">NAOMI (Deep Learning)</SelectItem>
                    <SelectItem value="TOTEM">TOTEM (VQVAE Tokenizer)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* KNN Parameters */}
              {model === "KNN" && (
                <>
                  <div className="space-y-2">
                    <label className="text-sm text-foreground">Neighbors (K)</label>
                    <Select value={String(nNeighbors)} onValueChange={(v) => setNNeighbors(Number(v))}>
                      <SelectTrigger className="bg-secondary/50 border-border/50">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {[1, 3, 5, 7, 10, 15, 20].map((k) => (
                          <SelectItem key={k} value={String(k)}>{k} neighbors</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  {showAdvanced && (
                    <>
                      <div className="space-y-2">
                        <label className="text-sm text-foreground">Weights</label>
                        <Select value={weights} onValueChange={(v) => setWeights(v as "uniform" | "distance")}>
                          <SelectTrigger className="bg-secondary/50 border-border/50">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="uniform">Uniform</SelectItem>
                            <SelectItem value="distance">Distance</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm text-foreground">Metric</label>
                        <Select value={metric} onValueChange={(v) => setMetric(v as "nan_euclidean" | "euclidean")}>
                          <SelectTrigger className="bg-secondary/50 border-border/50">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="nan_euclidean">NaN Euclidean</SelectItem>
                            <SelectItem value="euclidean">Euclidean</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </>
                  )}
                </>
              )}

              {/* MICE / REGRESSION Parameters */}
              {(model === "MICE" || model === "REGRESSION") && (
                <>
                  <div className="space-y-2">
                    <label className="text-sm text-foreground">Max Iterations</label>
                    <Select value={String(maxIter)} onValueChange={(v) => setMaxIter(Number(v))}>
                      <SelectTrigger className="bg-secondary/50 border-border/50">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="5">5 iterations</SelectItem>
                        <SelectItem value="10">10 iterations</SelectItem>
                        <SelectItem value="20">20 iterations</SelectItem>
                        <SelectItem value="50">50 iterations</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm text-foreground">Random State</label>
                    <Input
                      type="number"
                      value={randomState}
                      onChange={(e) => setRandomState(Number(e.target.value))}
                      className="bg-secondary/50 border-border/50"
                    />
                  </div>
                  {showAdvanced && (
                    <div className="space-y-2">
                      <label className="text-sm text-foreground">Estimator</label>
                      <Select value={estimator} onValueChange={(v) => setEstimator(v as typeof estimator)}>
                        <SelectTrigger className="bg-secondary/50 border-border/50">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="bayesian_ridge">Bayesian Ridge (default)</SelectItem>
                          <SelectItem value="random_forest">Random Forest</SelectItem>
                          <SelectItem value="extra_trees">Extra Trees</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  )}
                </>
              )}

              {/* NAOMI Parameters */}
              {model === "NAOMI" && (
                <>
                  <div className="space-y-2">
                    <label className="text-sm text-foreground">Epochs</label>
                    <Select value={String(naomiEpochs)} onValueChange={(v) => setNaomiEpochs(Number(v))}>
                      <SelectTrigger className="bg-secondary/50 border-border/50">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {[10, 25, 50, 100, 200].map((e) => (
                          <SelectItem key={e} value={String(e)}>{e} epochs</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  {showAdvanced && (
                    <>
                      <div className="space-y-2">
                        <label className="text-sm text-foreground">Hidden Dimension</label>
                        <Select value={String(hiddenDim)} onValueChange={(v) => setHiddenDim(Number(v))}>
                          <SelectTrigger className="bg-secondary/50 border-border/50">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {[32, 64, 128, 256].map((d) => (
                              <SelectItem key={d} value={String(d)}>{d}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm text-foreground">Learning Rate</label>
                        <Select value={String(naomiLr)} onValueChange={(v) => setNaomiLr(Number(v))}>
                          <SelectTrigger className="bg-secondary/50 border-border/50">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="0.01">0.01</SelectItem>
                            <SelectItem value="0.001">0.001 (default)</SelectItem>
                            <SelectItem value="0.0001">0.0001</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm text-foreground">Window Size</label>
                        <Select
                          value={String(naomiWindowSize)}
                          onValueChange={(v) => setNaomiWindowSize(Number(v))}
                        >
                          <SelectTrigger className="bg-secondary/50 border-border/50">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {[24, 50, 72, 96, 144].map((n) => (
                              <SelectItem key={n} value={String(n)}>{n} steps</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm text-foreground">Batch Size</label>
                        <Select
                          value={String(naomiBatchSize)}
                          onValueChange={(v) => setNaomiBatchSize(Number(v))}
                        >
                          <SelectTrigger className="bg-secondary/50 border-border/50">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {[16, 32, 64, 128].map((n) => (
                              <SelectItem key={n} value={String(n)}>{n}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm text-foreground">LSTM Layers</label>
                        <Select
                          value={String(naomiLayers)}
                          onValueChange={(v) => setNaomiLayers(Number(v))}
                        >
                          <SelectTrigger className="bg-secondary/50 border-border/50">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {[1, 2, 3, 4].map((n) => (
                              <SelectItem key={n} value={String(n)}>{n} layers</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm text-foreground">Highest Step (power of 2)</label>
                        <Select
                          value={String(naomiHighest)}
                          onValueChange={(v) => setNaomiHighest(Number(v))}
                        >
                          <SelectTrigger className="bg-secondary/50 border-border/50">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {[2, 4, 8, 16, 32].map((n) => (
                              <SelectItem key={n} value={String(n)}>{n}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm text-foreground">Gradient Clip</label>
                        <Select
                          value={String(naomiClip)}
                          onValueChange={(v) => setNaomiClip(Number(v))}
                        >
                          <SelectTrigger className="bg-secondary/50 border-border/50">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {[5, 10, 20].map((n) => (
                              <SelectItem key={n} value={String(n)}>{n}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm text-foreground">Preview Updates</label>
                        <Select
                          value={String(naomiPreviewUpdates)}
                          onValueChange={(v) => setNaomiPreviewUpdates(Number(v))}
                        >
                          <SelectTrigger className="bg-secondary/50 border-border/50">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {[5, 10, 20, 30, 50].map((n) => (
                              <SelectItem key={n} value={String(n)}>{n} updates</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </>
                  )}
                </>
              )}

              {/* TOTEM Parameters */}
              {model === "TOTEM" && (
                <>
                  <div className="space-y-2">
                    <label className="text-sm text-foreground">Window Size</label>
                    <Select value={String(totemWindowSize)} onValueChange={(v) => setTotemWindowSize(Number(v))}>
                      <SelectTrigger className="bg-secondary/50 border-border/50">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {[48, 96, 144, 192, 288].map((w) => (
                          <SelectItem key={w} value={String(w)}>{w} steps</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm text-foreground">Normalization</label>
                    <Select value={totemNormalization} onValueChange={(v) => setTotemNormalization(v as "zscore" | "minmax")}>
                      <SelectTrigger className="bg-secondary/50 border-border/50">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="zscore">Z-Score (default)</SelectItem>
                        <SelectItem value="minmax">Min-Max</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm text-foreground">Window Mode</label>
                    <Select
                      value={totemMergeMode}
                      onValueChange={(v) => setTotemMergeMode(v as "non_overlap" | "overlap")}
                    >
                      <SelectTrigger className="bg-secondary/50 border-border/50">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="non_overlap">Fast (Non-overlap)</SelectItem>
                        <SelectItem value="overlap">Smooth (Overlap)</SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-muted-foreground">
                      {totemMergeMode === "non_overlap"
                        ? "가장 빠른 모드입니다. 경계 품질보다 처리 속도를 우선합니다."
                        : "겹치는 구간을 평균 병합해 경계가 더 부드럽습니다. 대신 처리 시간이 늘어납니다."}
                    </p>
                  </div>
                  {totemMergeMode === "overlap" && (
                    <div className="space-y-2">
                      <label className="text-sm text-foreground">Stride</label>
                      <Select value={String(totemStride)} onValueChange={(v) => setTotemStride(Number(v))}>
                        <SelectTrigger className="bg-secondary/50 border-border/50">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {totemStrideOptions.map((s) => (
                            <SelectItem key={s} value={String(s)}>
                              {s} steps
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">
                        작을수록 더 부드럽지만 느려집니다. 1 ≤ stride &lt; window_size
                      </p>
                    </div>
                  )}
                  {showAdvanced && (
                    <div className="space-y-2">
                      <label className="text-sm text-foreground">Preview Updates</label>
                      <Select value={String(totemPreviewUpdates)} onValueChange={(v) => setTotemPreviewUpdates(Number(v))}>
                        <SelectTrigger className="bg-secondary/50 border-border/50">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {[5, 10, 20, 30].map((n) => (
                            <SelectItem key={n} value={String(n)}>{n} updates</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  )}
                </>
              )}

              {/* MEAN - no params */}
              {model === "MEAN" && (
                <p className="text-sm text-muted-foreground py-2">
                  No additional settings required.
                </p>
              )}

              {/* Advanced toggle (KNN / MICE / REGRESSION) */}
              {model !== "MEAN" && (
                <button
                  type="button"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                  <Settings2 className={`w-3 h-3 transition-transform ${showAdvanced ? "rotate-90" : ""}`} />
                  {showAdvanced ? "Hide advanced options" : "Show advanced options"}
                </button>
              )}

              <div className="pt-2 border-t border-border/50">
                <div className="flex items-start gap-2 text-xs text-muted-foreground">
                  <Info className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                  <p>
                    {model === "MICE" && "MICE uses chained equations to impute missing values iteratively."}
                    {model === "KNN" && "KNN uses nearest neighbors to estimate missing values."}
                    {model === "MEAN" && "Simple imputation using mean for numeric and mode for categorical."}
                    {model === "REGRESSION" && "Uses regression models to predict missing values."}
                    {model === "NAOMI" && "NAOMI uses a bidirectional LSTM with multiresolution decoding for sequence imputation."}
                    {model === "TOTEM" && "TOTEM uses a pretrained VQVAE tokenizer. Non-overlap is faster, overlap gives smoother boundaries with higher cost."}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Summary */}
          <Card className="border-border/50 bg-card/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Summary</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-primary">Target (impute)</span>
                <span className="text-foreground font-medium">{targetCount}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-blue-500">Feature (input only)</span>
                <span className="text-foreground font-medium">{featureCount}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Ignore</span>
                <span className="text-foreground font-medium">{ignoreCount}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Est. processing time</span>
                <span className="text-foreground font-medium">~2-5 min</span>
              </div>
              {targetCount === 0 && (
                <p className="text-xs text-destructive mt-1">Target column required</p>
              )}
            </CardContent>
          </Card>

          {/* Bulk Actions */}
          <Card className="border-border/50 bg-card/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Quick Actions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <button
                onClick={() => handleBulkRole("TARGET", (c) => c.nullRatio > 0)}
                className="w-full text-left text-xs px-3 py-2 rounded-md bg-secondary/50 hover:bg-secondary text-foreground transition-colors"
              >
                Missing &gt; 0 → Target
              </button>
              <button
                onClick={() => handleBulkRole("FEATURE")}
                className="w-full text-left text-xs px-3 py-2 rounded-md bg-secondary/50 hover:bg-secondary text-foreground transition-colors"
              >
                All → Feature
              </button>
              <button
                onClick={() => handleBulkRole("IGNORE", (c) => c.selectedType === "ID")}
                className="w-full text-left text-xs px-3 py-2 rounded-md bg-secondary/50 hover:bg-secondary text-foreground transition-colors"
              >
                ID columns → Ignore
              </button>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex items-center justify-between pt-4 border-t border-border/50">
        <Button variant="outline" onClick={onBack} className="gap-2">
          <ArrowLeft className="w-4 h-4" />
          Back
        </Button>
        <Button
          onClick={handleStartJob}
          disabled={targetCount === 0 || isLoading}
          className="gap-2 uppercase tracking-wide"
        >
          <Play className="w-4 h-4" />
          {isLoading ? "Starting..." : "Start Imputation"}
        </Button>
      </div>
    </div>
  )
}
