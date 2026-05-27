from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    upload_id: str | None = None
    sheet_name: str | None = None
    use_demo: bool = False
    dataset_name: str | None = None


class MethodRequest(BaseModel):
    method_id: str
    upload_id: str | None = None
    sheet_name: str | None = None
    use_demo: bool = False
    dataset_name: str | None = None
    params: dict = Field(default_factory=dict)


class TableRequest(BaseModel):
    upload_id: str | None = None
    sheet_name: str | None = None
    use_demo: bool = False
    dataset_name: str | None = None
    table_type: str = "baseline"
    group_var: str | None = None
    variables: list[str] | None = None
    decimal_places: int = 2
    p_digits: int = 3


class ExportRequest(BaseModel):
    upload_id: str | None = None
    format: str = "png"
    chart_data: dict | None = None
    table_data: list[dict] | None = None
