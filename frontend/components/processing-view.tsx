"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import {
  XCircle,
  Download,
  Plus,
  Clock,
  Cpu,
  CheckCircle2,
  Circle,
  Loader2,
  Eye,
  EyeOff,
  Zap,
} from "lucide-react"
import type { JobInfo } from "@/app/page"
import { useState } from "react"
import { ImputationChart } from "@/components/imputation-chart"

interface ProcessingViewProps {
  jobInfo: JobInfo
  onCancel: () => void
  onNewJob: () => void
  onDownload?: () => void
  onAugment?: () => void
  isComplete: boolean
}

const stages = ["Preprocess", "Encode", "Impute", "Export"]

export function ProcessingView({ jobInfo, onCancel, onNewJob, onDownload, onAugment, isComplete }: ProcessingViewProps) {
  const [showMissingOnly, setShowMissingOnly] = useState(true)
  const [visibleRows, setVisibleRows] = useState(200)

  return (
    <div className="space-y-6">
      {/* Job Info Header */}
      <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
        <CardContent className="p-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className={`w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-300 ${isComplete
                  ? "bg-primary/10 border border-primary/30 shadow-[0_0_20px_rgba(139,92,246,0.2)]"
                  : "bg-secondary border border-border"
                }`}>
                {isComplete ? (
                  <CheckCircle2 className="w-6 h-6 text-primary" />
                ) : (
                  <Loader2 className="w-6 h-6 text-primary animate-spin" />
                )}
              </div>
              <div>
                <div className="flex items-center gap-3">
                  <span className="text-sm text-muted-foreground">Job:</span>
                  <code className="text-sm font-mono text-foreground bg-secondary px-2 py-0.5 rounded">
                    {jobInfo.id}
                  </code>
                  <span className="text-sm text-muted-foreground">Model:</span>
                  <span className="text-sm text-foreground font-medium">{jobInfo.model}</span>
                </div>
                <div className="flex items-center gap-2 mt-1">
                  <span className={`text-xs px-2.5 py-0.5 rounded-md font-medium ${isComplete
                      ? "bg-primary/15 text-primary"
                      : "bg-warning/15 text-warning"
                    }`}>
                    {jobInfo.status}
                  </span>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Clock className="w-4 h-4" />
                <span>ETA: {jobInfo.eta}</span>
              </div>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="mt-6 space-y-3">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Progress</span>
              <span className="text-foreground font-medium">{Math.round(jobInfo.progress)}%</span>
            </div>
            <div className="relative">
              <Progress value={jobInfo.progress} className="h-2 bg-secondary" />
            </div>
          </div>

          {/* Stages */}
          <div className="mt-6 flex items-center justify-between">
            {stages.map((stage, index) => (
              <div key={stage} className="flex items-center gap-2">
                {index < jobInfo.stage ? (
                  <CheckCircle2 className="w-4 h-4 text-primary" />
                ) : index === jobInfo.stage ? (
                  <Loader2 className="w-4 h-4 text-primary animate-spin" />
                ) : (
                  <Circle className="w-4 h-4 text-muted-foreground/50" />
                )}
                <span className={`text-sm ${index <= jobInfo.stage ? "text-foreground" : "text-muted-foreground/50"
                  }`}>
                  {stage}
                </span>
                {index < stages.length - 1 && (
                  <div className={`hidden sm:block w-8 lg:w-16 h-px mx-2 ${index < jobInfo.stage ? "bg-primary" : "bg-border"
                    }`} />
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Imputation Chart */}
      <ImputationChart preview={jobInfo.imputationPreview} />

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Preview Panel */}
        <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
          <CardHeader className="pb-4">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base font-medium">Preview (sample rows only)</CardTitle>
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">rows:</span>
                <select
                  value={visibleRows}
                  onChange={(e) => setVisibleRows(Number(e.target.value))}
                  className="text-xs bg-secondary border border-border rounded px-2 py-1 text-foreground"
                >
                  <option value={100}>100</option>
                  <option value={200}>200</option>
                  <option value={500}>500</option>
                </select>
                <button
                  onClick={() => setShowMissingOnly(!showMissingOnly)}
                  className={`flex items-center gap-1 text-xs px-2 py-1 rounded transition-colors ${showMissingOnly
                      ? "bg-primary/10 text-primary border border-primary/20"
                      : "bg-secondary text-muted-foreground border border-border"
                    }`}
                >
                  {showMissingOnly ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
                  missing only
                </button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="rounded-lg border border-border overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-sm font-mono">
                  <thead>
                    <tr className="bg-secondary/50">
                      {jobInfo.previewData[0].map((header, i) => (
                        <th key={i} className="px-4 py-2 text-left text-muted-foreground font-medium border-b border-border">
                          {header}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {jobInfo.previewData.slice(1).map((row, rowIndex) => (
                      <tr key={rowIndex} className="border-b border-border/50 hover:bg-secondary/30">
                        {row.map((cell, cellIndex) => (
                          <td key={cellIndex} className="px-4 py-2">
                            {cell === null ? (
                              <span className={`px-2 py-0.5 rounded text-xs ${jobInfo.progress > 50 + rowIndex * 10
                                  ? "bg-primary/10 text-primary animate-pulse"
                                  : "bg-destructive/10 text-destructive"
                                }`}>
                                {jobInfo.progress > 50 + rowIndex * 10 ? "..." : "?"}
                              </span>
                            ) : (
                              <span className="text-foreground">{cell}</span>
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <p className="text-xs text-muted-foreground mt-3 text-center">
              * ? cells are being filled with imputed values
            </p>
          </CardContent>
        </Card>

        {/* Right Panel */}
        <div className="space-y-6">
          {/* What's happening */}
          <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
            <CardHeader className="pb-4">
              <CardTitle className="text-base font-medium flex items-center gap-2">
                <Cpu className="w-4 h-4 text-primary" />
                {"What's happening"}
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0 space-y-4">
              {/* Current Status */}
              <div className="space-y-2">
                {[
                  { text: "현재 범주형 원핫 인코딩 중", done: jobInfo.stage >= 1 },
                  { text: "결측치 마스크 생성", done: jobInfo.stage >= 2 },
                  { text: `MICE iter ${Math.min(10, Math.floor(jobInfo.progress / 10))}/10`, done: false },
                ].map((item, i) => (
                  <div key={i} className="flex items-center gap-2 text-sm">
                    <span className="text-muted-foreground">-</span>
                    <span className={item.done ? "text-primary" : "text-foreground"}>
                      {item.text}
                    </span>
                    {item.done && <CheckCircle2 className="w-3 h-3 text-primary" />}
                  </div>
                ))}
              </div>

              {/* Column fill progress */}
              <div className="space-y-3">
                <p className="text-sm text-muted-foreground">Column fill count (top)</p>
                {jobInfo.columns.map((col) => (
                  <div key={col.name} className="space-y-1">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-foreground font-mono">{col.name}</span>
                      <span className="text-muted-foreground">{Math.round(col.fillProgress)}%</span>
                    </div>
                    <div className="h-1.5 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary transition-all duration-300 shadow-[0_0_8px_rgba(139,92,246,0.5)]"
                        style={{ width: `${col.fillProgress}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Logs */}
          <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
            <CardHeader className="pb-4">
              <CardTitle className="text-base font-medium">Logs (last 20)</CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="h-40 overflow-y-auto rounded-lg bg-background/50 border border-border p-3 font-mono text-xs space-y-1">
                {jobInfo.logs.map((log, i) => (
                  <div key={i} className="text-muted-foreground">
                    {log}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex items-center justify-between">
        {isComplete ? (
          <>
            <Button
              variant="outline"
              onClick={onNewJob}
              className="gap-2"
            >
              <Plus className="w-4 h-4" />
              New Job
            </Button>
            <div className="flex items-center gap-3">
              <Button
                variant="outline"
                onClick={onAugment}
                className="gap-2 border-primary/30 text-primary hover:bg-primary/10 hover:border-primary/50"
              >
                <Zap className="w-4 h-4" />
                Augment (SMOTENC)
              </Button>
              <Button
                onClick={onDownload}
                className="gap-2 uppercase tracking-wide hover:shadow-[0_0_25px_rgba(139,92,246,0.4)]"
              >
                <Download className="w-4 h-4" />
                Download Result
              </Button>
            </div>
          </>
        ) : (
          <>
            <Button
              variant="outline"
              onClick={onCancel}
              className="gap-2 text-destructive hover:text-destructive border-destructive/30 hover:border-destructive/50 hover:bg-destructive/10"
            >
              <XCircle className="w-4 h-4" />
              Cancel
            </Button>
            <Button variant="outline" className="gap-2">
              View details
            </Button>
          </>
        )}
      </div>
    </div>
  )
}
