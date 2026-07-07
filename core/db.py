import json
from datetime import date, datetime

import pymysql
import pymysql.cursors

from nid_ocr.core.config import settings
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)

PAGE_SIZE = 50


def get_connection():
    return pymysql.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        database=settings.db_name,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def _ensure_database_exists() -> None:
    conn = pymysql.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        autocommit=True,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{settings.db_name}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
    finally:
        conn.close()


def init_db() -> None:
    """Create the database/table if they don't exist. Logs and continues on
    failure so a temporarily unreachable database doesn't block API startup."""
    try:
        _ensure_database_exists()
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ocr_uploads (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        side VARCHAR(10) NOT NULL,
                        original_filename VARCHAR(255) NOT NULL,
                        stored_path VARCHAR(500),
                        ocr_engine VARCHAR(50),
                        success TINYINT(1) NOT NULL,
                        extracted_data JSON,
                        error_message TEXT,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_side (side),
                        INDEX idx_success (success),
                        INDEX idx_created_at (created_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
        finally:
            conn.close()
        logger.info("Database ready (ocr_uploads table verified).")
    except Exception:
        logger.exception("Could not initialize database — /uploads history will be unavailable.")


def record_upload(
    side: str,
    original_filename: str,
    stored_path: str | None,
    ocr_engine: str,
    success: bool,
    extracted_data: dict | None = None,
    error_message: str | None = None,
) -> None:
    """Insert one row per processed upload. Never raises — a logging failure
    shouldn't affect the OCR response already computed for the caller."""
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ocr_uploads
                        (side, original_filename, stored_path, ocr_engine, success, extracted_data, error_message)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        side,
                        original_filename,
                        stored_path,
                        ocr_engine,
                        1 if success else 0,
                        json.dumps(extracted_data) if extracted_data is not None else None,
                        error_message,
                    ),
                )
        finally:
            conn.close()
    except Exception:
        logger.exception(f"Failed to record upload history for {original_filename}")


def query_uploads(
    side: str | None = None,
    status: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    search: str | None = None,
    page: int = 1,
) -> tuple[list[dict], int]:
    """Returns (rows, total_count) for the given filters, most recent first."""
    where = []
    params: list = []

    if side in ("front", "back"):
        where.append("side = %s")
        params.append(side)

    if status in ("success", "failed"):
        where.append("success = %s")
        params.append(1 if status == "success" else 0)

    if date_from:
        where.append("created_at >= %s")
        params.append(datetime.combine(date_from, datetime.min.time()))

    if date_to:
        where.append("created_at <= %s")
        params.append(datetime.combine(date_to, datetime.max.time()))

    if search:
        where.append("(original_filename LIKE %s OR JSON_SEARCH(extracted_data, 'one', %s) IS NOT NULL)")
        like = f"%{search}%"
        params.extend([like, like])

    where_clause = f"WHERE {' AND '.join(where)}" if where else ""
    offset = max(page - 1, 0) * PAGE_SIZE

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS total FROM ocr_uploads {where_clause}", params)
            total = cur.fetchone()["total"]

            cur.execute(
                f"""
                SELECT id, side, original_filename, stored_path, ocr_engine,
                       success, extracted_data, error_message, created_at
                FROM ocr_uploads
                {where_clause}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                params + [PAGE_SIZE, offset],
            )
            rows = cur.fetchall()
            for row in rows:
                if row["extracted_data"]:
                    row["extracted_data"] = json.loads(row["extracted_data"])
            return rows, total
    finally:
        conn.close()
