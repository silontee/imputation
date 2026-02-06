"use client"

import { useEffect, useMemo, useState } from "react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceDot,
} from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import type { ImputationPreview } from "@/lib/api"

interface ImputationChartProps {
  preview: ImputationPreview | null
}

interface ChartDataPoint {
  timestamp: string
  original: number | null
  imputed: number
  wasImputed: boolean
  combined: number // For continuous line
}

export function ImputationChart({ preview }: ImputationChartProps) {
  const [selectedDate, setSelectedDate] = useState<string>("")

  // Set or reset selected date when preview changes
  useEffect(() => {
    if (!preview?.dates_with_missing?.length) return
    if (!selectedDate || !preview.preview_data[selectedDate]) {
      setSelectedDate(preview.dates_with_missing[0])
    }
  }, [preview, selectedDate])

  const chartData = useMemo((): ChartDataPoint[] => {
    if (!preview || !selectedDate || !preview.preview_data[selectedDate]) {
      return []
    }

    const data = preview.preview_data[selectedDate]
    return data.timestamps.map((ts, i) => ({
      timestamp: ts.split(" ")[1] || ts, // Show only time part if available
      original: data.original[i],
      imputed: data.imputed[i],
      wasImputed: data.original[i] === null,
      combined: data.original[i] !== null ? data.original[i] : data.imputed[i],
    }))
  }, [preview, selectedDate])

  const stats = useMemo(() => {
    if (!chartData.length) return { total: 0, imputed: 0 }
    const imputed = chartData.filter((d) => d.wasImputed).length
    return { total: chartData.length, imputed }
  }, [chartData])

  const columnName = preview?.preview_data[selectedDate]?.column_name || "Value"

  if (!preview || !preview.dates_with_missing?.length) {
    return (
      <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base font-medium">Imputation Visualization</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[280px] flex items-center justify-center text-muted-foreground">
            No preview data available yet
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base font-medium">
            Imputation Visualization - {columnName}
          </CardTitle>
          <div className="flex items-center gap-4">
            <Select value={selectedDate} onValueChange={setSelectedDate}>
              <SelectTrigger className="w-40 h-8 text-xs bg-secondary/50 border-border/50">
                <SelectValue placeholder="Select date" />
              </SelectTrigger>
              <SelectContent>
                {preview.dates_with_missing.map((date) => (
                  <SelectItem key={date} value={date}>
                    {date}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        {/* Legend */}
        <div className="flex items-center gap-6 mb-4 text-xs">
          <div className="flex items-center gap-1.5">
            <div className="w-5 h-0.5 bg-primary rounded" />
            <span className="text-muted-foreground">Original Data</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="flex items-center">
              <div className="w-5 h-0.5 bg-chart-2 rounded" style={{ backgroundImage: "repeating-linear-gradient(90deg, hsl(var(--chart-2)) 0px, hsl(var(--chart-2)) 4px, transparent 4px, transparent 7px)" }} />
              <div className="w-2 h-2 rounded-full bg-chart-2 -ml-0.5" />
            </div>
            <span className="text-muted-foreground">Imputed Values ({stats.imputed})</span>
          </div>
        </div>

        {/* Chart */}
        <div className="h-[280px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(var(--border))"
                strokeOpacity={0.5}
              />
              <XAxis
                dataKey="timestamp"
                tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }}
                axisLine={{ stroke: "hsl(var(--border))" }}
                tickLine={{ stroke: "hsl(var(--border))" }}
                interval="preserveStartEnd"
              />
              <YAxis
                tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }}
                axisLine={{ stroke: "hsl(var(--border))" }}
                tickLine={{ stroke: "hsl(var(--border))" }}
                width={50}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "8px",
                  boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
                }}
                labelStyle={{ color: "hsl(var(--foreground))" }}
                formatter={(value: number, name: string, props) => {
                  const point = props.payload as ChartDataPoint
                  if (name === "imputed" && point.wasImputed) {
                    return [value?.toFixed(2), "Imputed"]
                  }
                  if (name === "combined" && !point.wasImputed) {
                    return [value?.toFixed(2), "Original"]
                  }
                  return [null, ""]
                }}
              />

              {/* Base continuous line (original + imputed connected) */}
              <Line
                type="monotone"
                dataKey="combined"
                stroke="hsl(var(--primary))"
                strokeWidth={2}
                dot={false}
                connectNulls={true}
                name="combined"
              />

              {/* Overlay: imputed segments with different style */}
              <Line
                type="monotone"
                dataKey={(d: ChartDataPoint) => (d.wasImputed ? d.combined : null)}
                stroke="hsl(var(--chart-2))"
                strokeWidth={3}
                strokeDasharray="6 4"
                dot={{
                  fill: "hsl(var(--chart-2))",
                  stroke: "hsl(var(--background))",
                  strokeWidth: 2,
                  r: 5,
                }}
                connectNulls={true}
                name="imputed"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Info */}
        <div className="mt-3 text-xs text-muted-foreground text-center">
          Showing {stats.total} data points for {selectedDate} • {stats.imputed} values imputed
        </div>
      </CardContent>
    </Card>
  )
}
