export type ColumnType = "ID" | "NUMERIC" | "CATEGORICAL" | "DATETIME"
export type ColumnAction = "IMPUTE" | "IGNORE"
export type ColumnRole = "TARGET" | "FEATURE" | "IGNORE"

export interface ColumnProfile {
  name: string
  detectedType: ColumnType
  nullCount: number
  nullRatio: number
  uniqueCount: number
  example: (string | number | null)[]
  recommendedAction: ColumnAction
  warnings: string[]
  // User-modifiable fields
  selectedType: ColumnType
  selectedRole: ColumnRole
}

export interface SchemaReviewData {
  jobId: string
  filename: string
  sampleRows: number
  totalNullRatio: number
  columns: ColumnProfile[]
}

export interface ModelConfig {
  type: "MICE" | "KNN" | "MEAN" | "REGRESSION" | "NAOMI" | "TOTEM"
  params: {
    // MICE / REGRESSION
    maxIter?: number
    randomState?: number
    estimator?: "bayesian_ridge" | "random_forest" | "extra_trees"
    // KNN
    nNeighbors?: number
    weights?: "uniform" | "distance"
    metric?: "nan_euclidean" | "euclidean"
    // NAOMI
    hiddenDim?: number
    epochs?: number
    lr?: number
    numResolutions?: number | null
    highest?: number
    windowSize?: number
    batchSize?: number
    nLayers?: number
    clip?: number
    previewUpdates?: number
    // TOTEM
    totemWindowSize?: number
    totemNormalization?: "zscore" | "minmax"
    totemMergeMode?: "non_overlap" | "overlap"
    totemStride?: number
    totemPreviewUpdates?: number
  }
}

export interface JobConfig {
  model: ModelConfig
  columns: {
    name: string
    type: ColumnType
    role: ColumnRole
  }[]
}
