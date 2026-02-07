from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

from app.core.config import settings
from app.schemas.analyze import AnalyzeResponse
from app.services.analyzer import analyze_dataframe
from app.services.csv_reader import read_csv_with_fallback, ENCODING_CANDIDATES
from app.db.session import get_db
from app.db.repository import JobRepository

router = APIRouter(tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a CSV file and analyze its schema."""

    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported")

    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    # Check file size
    size_bytes = len(content)
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_MB}MB"
        )

    repo = JobRepository(db)

    # Create job record
    job = repo.create(
        filename=file.filename,
        input_path="",
        file_size_bytes=size_bytes,
    )
    job_id = str(job.id)

    # Save file to uploads directory
    file_path = settings.UPLOAD_DIR / f"{job_id}_input.csv"
    try:
        with open(file_path, "wb") as f:
            f.write(content)
        repo.update(job_id, input_path=str(file_path))
    except Exception as e:
        repo.delete(job_id)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Read sample rows and analyze
    try:
        df, enc = read_csv_with_fallback(content, nrows=settings.SAMPLE_ROWS)
        columns = analyze_dataframe(df)
    except Exception as e:
        repo.delete(job_id)
        if isinstance(e, (UnicodeDecodeError, ValueError)):
            tried = ", ".join(ENCODING_CANDIDATES)
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse CSV with supported encodings ({tried}): {str(e)}",
            )
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    # Calculate total null ratio
    total_nulls = sum(col.null_count for col in columns)
    total_cells = len(df) * len(df.columns)
    total_null_ratio = total_nulls / max(total_cells, 1)

    repo.add_log(job_id, f"File uploaded: {file.filename}")
    repo.add_log(job_id, f"CSV encoding detected: {enc}")
    repo.add_log(job_id, "Missing values normalized: 'NA'")
    repo.add_log(job_id, f"Analyzed {len(df)} rows, {len(columns)} columns")

    return AnalyzeResponse(
        job_id=job_id,
        filename=file.filename,
        sample_rows=len(df),
        total_null_ratio=round(total_null_ratio, 4),
        columns=columns,
    )
