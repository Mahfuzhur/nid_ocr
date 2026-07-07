from datetime import date, datetime
from html import escape
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse

from nid_ocr.core.db import query_uploads, PAGE_SIZE, get_connection
from nid_ocr.core.logging import get_logger

logger = get_logger(__name__)


class UploadsRouter:
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route(
            "/uploads",
            self._list,
            methods=["GET"],
            summary="Browse uploaded images and their extracted data",
            include_in_schema=False,
        )
        self.router.add_api_route(
            "/uploads/{record_id}/image",
            self._image,
            methods=["GET"],
            summary="Fetch the stored image for an upload record",
            include_in_schema=False,
        )

    async def _list(
        self,
        side: str | None = Query(None),
        status: str | None = Query(None),
        date_from: date | None = Query(None),
        date_to: date | None = Query(None),
        search: str | None = Query(None),
        page: int = Query(1, ge=1),
    ) -> HTMLResponse:
        rows, total = query_uploads(
            side=side, status=status, date_from=date_from, date_to=date_to, search=search, page=page,
        )
        return HTMLResponse(self._render(rows, total, side, status, date_from, date_to, search, page))

    async def _image(self, record_id: int):
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT stored_path FROM ocr_uploads WHERE id = %s", (record_id,))
                row = cur.fetchone()
        finally:
            conn.close()
        if not row or not row["stored_path"]:
            raise HTTPException(status_code=404, detail="Image not found.")
        return FileResponse(row["stored_path"])

    def _render(self, rows, total, side, status, date_from, date_to, search, page) -> str:
        filters = {
            "side": side or "",
            "status": status or "",
            "date_from": date_from.isoformat() if date_from else "",
            "date_to": date_to.isoformat() if date_to else "",
            "search": search or "",
        }

        def option(value: str, label: str, current: str) -> str:
            selected = " selected" if value == current else ""
            return f'<option value="{value}"{selected}>{label}</option>'

        table_rows = "\n".join(self._render_row(r) for r in rows) or (
            '<tr><td colspan="7" class="empty">No uploads match these filters.</td></tr>'
        )

        total_pages = max((total + PAGE_SIZE - 1) // PAGE_SIZE, 1)
        pagination = self._render_pagination(filters, page, total_pages)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NID OCR — Uploaded Images</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #f7f8fa; color: #1a1a1a; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 1rem; }}
  form.filters {{ display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: end; margin-bottom: 1.5rem;
                  background: #fff; padding: 1rem; border-radius: 8px; border: 1px solid #e0e0e0; }}
  form.filters label {{ display: flex; flex-direction: column; font-size: 0.8rem; color: #555; gap: 0.25rem; }}
  form.filters input, form.filters select {{ padding: 0.35rem 0.5rem; border: 1px solid #ccc; border-radius: 4px; }}
  form.filters button {{ padding: 0.45rem 1rem; border: none; border-radius: 4px; background: #2563eb; color: #fff; cursor: pointer; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; }}
  th, td {{ padding: 0.6rem 0.75rem; border-bottom: 1px solid #eee; text-align: left; vertical-align: top; font-size: 0.85rem; }}
  th {{ background: #f0f2f5; }}
  tr:last-child td {{ border-bottom: none; }}
  .badge {{ padding: 0.15rem 0.5rem; border-radius: 999px; font-size: 0.75rem; }}
  .badge.success {{ background: #dcfce7; color: #166534; }}
  .badge.failed {{ background: #fee2e2; color: #991b1b; }}
  .empty {{ text-align: center; color: #888; padding: 2rem; }}
  .fields dt {{ font-weight: 600; display: inline; }}
  .fields dd {{ display: inline; margin: 0 0.5rem 0 0.25rem; }}
  .fields div {{ margin-bottom: 0.15rem; }}
  .pagination {{ margin-top: 1rem; display: flex; gap: 1rem; align-items: center; font-size: 0.85rem; }}
  .pagination a {{ color: #2563eb; text-decoration: none; }}
  .thumb-wrap {{ position: relative; display: inline-block; vertical-align: middle; cursor: zoom-in; }}
  .thumb-toggle {{ display: none; }}
  .thumb {{ width: 44px; height: 44px; object-fit: cover; border-radius: 4px; border: 1px solid #ddd;
            transition: all 0.15s ease; }}
  .thumb-toggle:checked ~ .thumb {{
    width: 340px; height: 340px; object-fit: contain; background: #fff;
    position: absolute; top: 50%; left: 0; transform: translateY(-50%);
    z-index: 20; border-radius: 6px; box-shadow: 0 6px 20px rgba(0,0,0,0.35);
    cursor: zoom-out;
  }}
</style>
</head>
<body>
<h1>Uploaded NID Images ({total} total)</h1>

<form class="filters" method="get" action="/uploads">
  <label>Side
    <select name="side">
      {option("", "All", filters["side"])}
      {option("front", "Front", filters["side"])}
      {option("back", "Back", filters["side"])}
    </select>
  </label>
  <label>Status
    <select name="status">
      {option("", "All", filters["status"])}
      {option("success", "Success", filters["status"])}
      {option("failed", "Failed", filters["status"])}
    </select>
  </label>
  <label>From
    <input type="date" name="date_from" value="{filters['date_from']}">
  </label>
  <label>To
    <input type="date" name="date_to" value="{filters['date_to']}">
  </label>
  <label>Search
    <input type="text" name="search" placeholder="filename, name, NID number..." value="{escape(filters['search'])}">
  </label>
  <button type="submit">Filter</button>
</form>

<table>
  <thead>
    <tr>
      <th>ID</th><th>Side</th><th>Filename</th><th>Uploaded</th><th>Engine</th><th>Status</th><th>Extracted Data</th>
    </tr>
  </thead>
  <tbody>
    {table_rows}
  </tbody>
</table>

{pagination}
</body>
</html>"""

    @staticmethod
    def _render_row(row: dict) -> str:
        created = row["created_at"]
        created_s = created.strftime("%Y-%m-%d %H:%M") if isinstance(created, datetime) else escape(str(created))

        if row["success"]:
            badge = '<span class="badge success">Success</span>'
            data = row["extracted_data"] or {}
            fields = "".join(
                f"<div><dt>{escape(str(k))}:</dt><dd>{escape(str(v)) if v is not None else '—'}</dd></div>"
                for k, v in data.items()
            )
            details = f'<dl class="fields">{fields}</dl>' if fields else "—"
        else:
            badge = '<span class="badge failed">Failed</span>'
            details = escape(row["error_message"] or "Unknown error")

        image_link = (
            f'<label class="thumb-wrap">'
            f'<input type="checkbox" class="thumb-toggle">'
            f'<img class="thumb" src="/uploads/{row["id"]}/image" alt="preview" loading="lazy"></label>'
            if row["stored_path"] else "—"
        )

        return f"""<tr>
      <td>{row['id']}</td>
      <td>{escape(row['side'])}</td>
      <td>{escape(row['original_filename'])} {image_link}</td>
      <td>{created_s}</td>
      <td>{escape(row['ocr_engine'] or '')}</td>
      <td>{badge}</td>
      <td>{details}</td>
    </tr>"""

    @staticmethod
    def _render_pagination(filters: dict, page: int, total_pages: int) -> str:
        def link(p: int, label: str) -> str:
            qs = urlencode({**{k: v for k, v in filters.items() if v}, "page": p})
            return f'<a href="/uploads?{qs}">{label}</a>'

        prev_link = link(page - 1, "← Previous") if page > 1 else "← Previous"
        next_link = link(page + 1, "Next →") if page < total_pages else "Next →"
        return f'<div class="pagination">{prev_link} <span>Page {page} of {total_pages}</span> {next_link}</div>'
