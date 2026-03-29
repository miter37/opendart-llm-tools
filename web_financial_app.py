from __future__ import annotations

import json
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from opendart_llm_tools.opendart_llm_tools import (
    OpenDartReportExplorer,
    _load_key_env,
    _resolve_gemini_api_keys,
)

WEB_DIR = BASE_DIR / "web"
CSS_PATH = BASE_DIR / "web_app.css"
JS_PATH = BASE_DIR / "web_app.js"


class CompanySearchRequest(BaseModel):
    query: str


class ReportsRequest(BaseModel):
    corp_code: str


class AnalyzeRequest(BaseModel):
    corp_code: str
    company_name: str
    rcept_no: str
    provider: str = "gemini"
    statement_scope: str = "consolidated"


@lru_cache(maxsize=1)
def get_key_env() -> Dict[str, str]:
    return _load_key_env(str(BASE_DIR / "key.env"))


@lru_cache(maxsize=1)
def get_explorer() -> OpenDartReportExplorer:
    key_env = get_key_env()
    dart_key = key_env.get("opendart_key")
    if not dart_key:
        raise RuntimeError("opendart_key not found in key.env")
    return OpenDartReportExplorer(dart_key)


def compact(text: Any) -> str:
    return "".join(str(text or "").split())


def parse_amount(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "-":
        return None
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]
    text = text.replace(",", "")
    try:
        number = int(float(text))
    except ValueError:
        return None
    return -number if negative else number


def format_krw_short(value: Optional[int]) -> str:
    if value is None:
        return "-"
    negative = value < 0
    value = abs(value)
    cho = 1_000_000_000_000
    eok = 100_000_000
    if value >= cho:
        cho_val = value // cho
        eok_val = (value % cho) // eok
        text = f"{cho_val}조 {eok_val:,}억원" if eok_val else f"{cho_val}조원"
    elif value >= eok:
        text = f"{value / eok:,.0f}억원"
    else:
        text = f"{value:,}원"
    return f"-{text}" if negative else text


def row_label(row: List[Any]) -> str:
    texts = [str(cell) for cell in row if isinstance(cell, str) and str(cell).strip()]
    return compact(" ".join(texts))


def row_last_amount(row: List[Any]) -> Optional[int]:
    for cell in reversed(row):
        amount = parse_amount(cell)
        if amount is not None:
            return amount
    return None


def find_company_candidates(query: str) -> List[Dict[str, Any]]:
    query_norm = (query or "").strip()
    if not query_norm:
        return []

    explorer = get_explorer()
    df = explorer.get_corp_codes()
    exact = df[df["corp_name"] == query_norm]
    if not exact.empty:
        return exact.head(1).to_dict(orient="records")

    contains = df[
        df["corp_name"].str.contains(query_norm, case=False, na=False)
        | df["stock_code"].str.contains(query_norm, case=False, na=False)
    ]
    if contains.empty:
        return []
    return contains.head(12).to_dict(orient="records")


def list_recent_reports(corp_code: str) -> List[Dict[str, Any]]:
    explorer = get_explorer()
    today = date.today()
    start_year = today.year - 5
    filings = explorer.search_filings(
        corp_code=corp_code,
        bgn_de=f"{start_year}0101",
        end_de=today.strftime("%Y%m%d"),
        pblntf_detail_ty=None,
        last_reprt_at="Y",
        report_type="regular",
    )
    if filings.empty:
        return []
    return filings.to_dict(orient="records")[:16]


def _select_statement_amount(
    rows: List[Dict[str, Any]],
    exact_names: List[str],
    sj_divs: Optional[List[str]] = None,
    sum_matches: bool = True,
) -> int:
    exact_compact = {compact(name) for name in exact_names}
    matches: List[int] = []
    for row in rows:
        if sj_divs and row.get("sj_div") not in sj_divs:
            continue
        if compact(row.get("account_nm")) in exact_compact:
            amount = parse_amount(row.get("thstrm_amount"))
            if amount is not None:
                matches.append(amount)
    if not matches:
        return 0
    if sum_matches:
        return sum(matches)
    return matches[0]


def _select_statement_amount_by_ids(
    rows: List[Dict[str, Any]],
    account_ids: List[str],
    sj_divs: Optional[List[str]] = None,
    sum_matches: bool = True,
) -> int:
    id_set = set(account_ids)
    matches: List[int] = []
    for row in rows:
        if sj_divs and row.get("sj_div") not in sj_divs:
            continue
        if row.get("account_id") in id_set:
            amount = parse_amount(row.get("thstrm_amount"))
            if amount is not None:
                matches.append(amount)
    if not matches:
        return 0
    if sum_matches:
        return sum(matches)
    return matches[0]


def _first_nonzero_amount(*values: Optional[int]) -> Optional[int]:
    for value in values:
        if value:
            return value
    return None


def _extract_note_section(
    explorer: OpenDartReportExplorer,
    note_page: Dict[str, Any],
    sections: List[Dict[str, Any]],
    keywords: List[str],
) -> List[Dict[str, Any]]:
    matched = explorer._match_section_from_toc(sections, keywords)
    if not matched:
        return []
    extracted = explorer.extract_report_section(
        url=note_page["url"],
        page_title=note_page["title"],
        toc_id=matched["toc_id"],
        max_table_rows=30,
        max_blocks=20,
    )
    return extracted.get("blocks", [])


def build_snapshot(corp_code: str, rcept_no: str, statement_scope: str) -> Dict[str, Any]:
    explorer = get_explorer()
    filing = explorer._find_filing_by_rcept_no(corp_code, rcept_no)
    report_nm = filing["report_nm"]
    bsns_year = explorer._extract_business_year(report_nm)
    reprt_code = explorer._reprt_code_from_report_nm(report_nm)
    fs_div, scope_label = explorer._fs_div_from_scope(statement_scope)
    rows_df = explorer._fetch_financial_statement_rows(corp_code, bsns_year, reprt_code, fs_div)
    rows = rows_df.to_dict(orient="records")

    revenue = _first_nonzero_amount(
        _select_statement_amount_by_ids(rows, ["ifrs-full_Revenue"], sj_divs=["IS", "CIS"], sum_matches=False),
        _select_statement_amount(rows, ["매출액"], sj_divs=["IS", "CIS"], sum_matches=False),
    ) or 0
    operating_income = _first_nonzero_amount(
        _select_statement_amount_by_ids(
            rows,
            ["dart_OperatingIncomeLoss", "ifrs-full_ProfitLossFromOperatingActivities"],
            sj_divs=["IS", "CIS"],
            sum_matches=False,
        ),
        _select_statement_amount(rows, ["영업이익", "영업손익", "Ⅴ.영업이익"], sj_divs=["IS", "CIS"], sum_matches=False),
    ) or 0
    net_income = _first_nonzero_amount(
        _select_statement_amount_by_ids(rows, ["ifrs-full_ProfitLoss"], sj_divs=["IS", "CIS"], sum_matches=False),
        _select_statement_amount(
            rows,
            ["당기순이익", "당기순이익(손실)", "당기순손익", "XI. 당기순이익"],
            sj_divs=["IS", "CIS"],
            sum_matches=False,
        ),
    ) or 0
    equity = _first_nonzero_amount(
        _select_statement_amount_by_ids(rows, ["ifrs-full_Equity"], sj_divs=["BS"], sum_matches=False),
        _select_statement_amount(rows, ["자본총계", "자본 총계"], sj_divs=["BS"], sum_matches=False),
    ) or 0
    cash = _first_nonzero_amount(
        _select_statement_amount_by_ids(rows, ["ifrs-full_CashAndCashEquivalents"], sj_divs=["BS"], sum_matches=False),
        _select_statement_amount(rows, ["현금및현금성자산"], sj_divs=["BS"], sum_matches=False),
    ) or 0
    deposits = _first_nonzero_amount(
        _select_statement_amount_by_ids(
            rows,
            [
                "ifrs-full_ShorttermDepositsNotClassifiedAsCashEquivalents",
                "dart_ShortTermFinancialInstitutionDeposits",
                "dart_LongTermFinancialInstitutionDeposits",
            ],
            sj_divs=["BS"],
        ),
        _select_statement_amount(
            rows,
            ["단기금융기관예치금", "장기금융기관예치금", "단기금융상품", "장기금융상품"],
            sj_divs=["BS"],
        ),
    ) or 0

    debt_by_ids = (
        _select_statement_amount_by_ids(
            rows,
            [
                "ifrs-full_CurrentBorrowings",
                "ifrs-full_CurrentPortionOfLongTermBorrowings",
                "ifrs-full_CurrentPortionOfLongTermBondsIssued",
                "ifrs-full_NoncurrentBorrowings",
                "ifrs-full_BondsIssued",
                "ifrs-full_LeaseLiabilities",
                "ifrs-full_CurrentLeaseLiabilities",
            ],
            sj_divs=["BS"],
        )
        + _select_statement_amount(rows, ["교환사채", "전환사채", "신주인수권부사채"], sj_divs=["BS"])
        + _select_statement_amount(
            rows,
            ["유동 차입금 및 비유동차입금(사채 포함)의 유동성 대체 부분"],
            sj_divs=["BS"],
        )
    )
    debt_by_names = _select_statement_amount(
        rows,
        [
            "단기차입금",
            "단기사채",
            "유동성사채",
            "유동 차입금 및 비유동차입금(사채 포함)의 유동성 대체 부분",
            "장기차입금",
            "사채",
            "교환사채",
            "전환사채",
            "신주인수권부사채",
            "리스부채",
        ],
        sj_divs=["BS"],
    )
    debt = max(debt_by_ids, debt_by_names)

    net_debt = None
    if any(value is not None and value != 0 for value in [debt, cash, deposits]):
        net_debt = (debt or 0) - (cash or 0) - (deposits or 0)

    pages = explorer.get_all_report_pages(rcept_no)
    note_kind = "note_consolidated" if scope_label == "연결" else "note_separate"
    note_page = next((page for page in pages if page.get("kind") == note_kind), None)
    sections = explorer.list_page_subsections(note_page["url"], note_page["title"]) if note_page else []

    depreciation = None
    depreciation_with_rou = None
    receivables = _first_nonzero_amount(
        _select_statement_amount_by_ids(
            rows,
            ["ifrs-full_CurrentTradeReceivables", "ifrs-full_TradeReceivables"],
            sj_divs=["BS"],
            sum_matches=False,
        ),
        _select_statement_amount(rows, ["매출채권"], sj_divs=["BS"], sum_matches=False),
        _select_statement_amount(rows, ["매출채권및기타채권", "매출채권 및 기타채권"], sj_divs=["BS"], sum_matches=False),
        _select_statement_amount(rows, ["장기매출채권"], sj_divs=["BS"], sum_matches=False),
    )
    if note_page:
        depreciation_blocks = _extract_note_section(
            explorer,
            note_page,
            sections,
            ["비용의 성격별 분류", "비용의 성격", "영업비용"],
        )
        dep_core = 0
        dep_rou = 0
        for block in depreciation_blocks:
            if block["type"] != "table":
                continue
            for row in block["content"].get("rows", []):
                label = row_label(row)
                amount = row_last_amount(row)
                if amount is None:
                    continue
                if "사용권자산상각비" in label:
                    dep_rou = amount
                elif "감가상각비" in label:
                    dep_core = amount
        if dep_core:
            depreciation = dep_core * 1000
            depreciation_with_rou = (dep_core + dep_rou) * 1000

        if receivables is None:
            receivable_blocks = _extract_note_section(
                explorer,
                note_page,
                sections,
                ["매출채권및기타채권", "매출채권 및 기타채권", "매출채권"],
            )
            receivable_total = 0
            for block in receivable_blocks:
                if block["type"] != "table":
                    continue
                for row in block["content"].get("rows", []):
                    label = row_label(row)
                    amount = row_last_amount(row)
                    if amount is None:
                        continue
                    if "유동매출채권" in label or "비유동매출채권" in label:
                        receivable_total += amount
            if receivable_total:
                receivables = receivable_total * 1000

    return {
        "report_nm": report_nm,
        "scope_label": scope_label,
        "revenue": revenue,
        "operating_income": operating_income,
        "net_income": net_income,
        "equity": equity,
        "cash": cash,
        "debt": debt,
        "net_debt": net_debt,
        "receivables": receivables,
        "depreciation": depreciation,
        "depreciation_with_rou": depreciation_with_rou,
        "cards": [
            {"label": "매출", "value": format_krw_short(revenue)},
            {"label": "영업이익", "value": format_krw_short(operating_income)},
            {"label": "순이익", "value": format_krw_short(net_income)},
            {"label": "자본총계", "value": format_krw_short(equity)},
            {"label": "순부채", "value": format_krw_short(net_debt)},
            {"label": "매출채권", "value": format_krw_short(receivables)},
        ],
    }


def gemini_text_completion(prompt: str, model: str = "gemini-2.0-flash") -> str:
    keys = _resolve_gemini_api_keys(get_key_env())
    if not keys:
        raise RuntimeError("Gemini API key not found.")

    payload = {
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "너는 CFA 스타일의 재무분석가다. "
                        "반드시 수치 근거를 인용하고, 제공된 자료에 없는 수치는 추정이라고 표시하라."
                    )
                }
            ]
        },
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
    }

    last_error: Optional[str] = None
    for api_key in keys:
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": api_key},
            json=payload,
            timeout=120,
        )
        if resp.ok:
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError(f"Gemini returned no candidates: {data}")
            parts = candidates[0].get("content", {}).get("parts", [])
            return "\n".join(part.get("text", "") for part in parts if part.get("text")).strip()

        last_error = resp.text
        if resp.status_code not in {403, 429}:
            resp.raise_for_status()

    raise RuntimeError(f"Gemini request failed: {last_error}")


def openai_text_completion(prompt: str, model: str = "gpt-5.4") -> str:
    if OpenAI is None:
        raise RuntimeError("openai package is not available.")

    api_key = get_key_env().get("openai_key")
    if not api_key:
        raise RuntimeError("openai_key not found in key.env")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=(
            "너는 CFA 스타일의 재무분석가다. "
            "반드시 수치 근거를 인용하고, 제공된 자료에 없는 수치는 추정이라고 표시하라."
        ),
        input=prompt,
    )
    return response.output_text


def analyze_with_llm(provider: str, company_name: str, snapshot: Dict[str, Any], markdown_text: str) -> str:
    prompt = f"""
회사명: {company_name}
보고서명: {snapshot['report_nm']}
재무제표 구분: {snapshot['scope_label']}

핵심 숫자 스냅샷:
{json.dumps(snapshot, ensure_ascii=False, indent=2)}

요청:
1. 매출, 영업이익, 순이익, 자본, 순부채 같은 핵심 숫자를 짧게 요약
2. 수익성, 레버리지, 유동성, 현금흐름 관점의 중요한 분석 포인트 제시
3. 감가상각, 매출채권, 재고, 차입금, 비용구조 변화가 중요하면 꼭 언급
4. 투자자 또는 애널리스트가 추가로 확인할 질문 5개 제시
5. 결과는 Markdown으로 작성

원문 자료:
{markdown_text}
""".strip()

    provider_norm = (provider or "gemini").strip().lower()
    if provider_norm == "openai":
        return openai_text_completion(prompt)
    return gemini_text_completion(prompt)


app = FastAPI(title="OpenDART Analyst")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/assets/app.css")
def app_css() -> FileResponse:
    return FileResponse(CSS_PATH)


@app.get("/assets/app.js")
def app_js() -> FileResponse:
    return FileResponse(JS_PATH)


@app.post("/api/company-search")
def company_search(body: CompanySearchRequest):
    candidates = find_company_candidates(body.query)
    if not candidates:
        raise HTTPException(status_code=404, detail="회사를 찾지 못했습니다.")
    return {"candidates": candidates}


@app.post("/api/reports")
def reports(body: ReportsRequest):
    items = list_recent_reports(body.corp_code)
    if not items:
        raise HTTPException(status_code=404, detail="조회 가능한 정기보고서를 찾지 못했습니다.")
    return {"reports": items}


@app.post("/api/analyze")
def analyze(body: AnalyzeRequest):
    explorer = get_explorer()
    markdown_text = explorer.get_financial_statement_key_contents(
        corp_code=body.corp_code,
        rcept_no=body.rcept_no,
        statement_scope=body.statement_scope,
        max_table_rows=12,
        max_note_blocks=18,
    )
    snapshot = build_snapshot(body.corp_code, body.rcept_no, body.statement_scope)
    analysis = analyze_with_llm(body.provider, body.company_name, snapshot, markdown_text)
    return {
        "snapshot": snapshot,
        "markdown": markdown_text,
        "analysis": analysis,
    }
