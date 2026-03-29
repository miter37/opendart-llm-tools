import argparse
import calendar
import datetime as dt
import io
import os
import re
import json
import random
import time
import uuid
import zipfile
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict
from urllib.parse import urlencode

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

OPENDART_API = "https://opendart.fss.or.kr/api"
DART_WEB = "https://dart.fss.or.kr"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

REGULAR_REPRT_CODE_LABELS = {
    "11013": "1분기보고서",
    "11012": "반기보고서",
    "11014": "3분기보고서",
    "11011": "사업보고서",
}

PBLNTF_TYPE_LABELS = {
    "A": "정기공시",
    "B": "주요사항보고",
    "C": "발행공시",
    "D": "지분공시",
    "E": "기타공시",
    "F": "외부감사관련",
    "G": "펀드공시",
    "H": "자산유동화",
    "I": "거래소공시",
    "J": "공정위공시",
}

PBLNTF_DETAIL_TYPE_LABELS = {
    "A001": "사업보고서",
    "A002": "반기보고서",
    "A003": "분기보고서",
    "A004": "등록법인결산서류(자본시장법이전)",
    "A005": "소액공모법인결산서류",
    "B001": "주요사항보고서",
    "B002": "주요경영사항신고(자본시장법 이전)",
    "B003": "최대주주등과의거래신고(자본시장법 이전)",
    "C001": "증권신고(지분증권)",
    "C002": "증권신고(채무증권)",
    "C003": "증권신고(파생결합증권)",
    "C004": "증권신고(합병등)",
    "C005": "증권신고(기타)",
    "C006": "소액공모(지분증권)",
    "C007": "소액공모(채무증권)",
    "C008": "소액공모(파생결합증권)",
    "C009": "소액공모(합병등)",
    "C010": "소액공모(기타)",
    "C011": "호가중개시스템을통한소액매출",
    "D001": "주식등의대량보유상황보고서",
    "D002": "임원ㆍ주요주주특정증권등소유상황보고서",
    "D003": "의결권대리행사권유",
    "D004": "공개매수",
    "E001": "자기주식취득/처분",
    "E002": "신탁계약체결/해지",
    "E003": "합병등종료보고서",
    "E004": "주식매수선택권부여에관한신고",
    "E005": "사외이사에관한신고",
    "E006": "주주총회소집공고",
    "E007": "시장조성/안정조작",
    "E008": "합병등신고서(자본시장법 이전)",
    "E009": "금융위등록/취소(자본시장법 이전)",
    "F001": "감사보고서",
    "F002": "연결감사보고서",
    "F003": "결합감사보고서",
    "F004": "회계법인사업보고서",
    "F005": "감사전재무제표미제출신고서",
    "G001": "증권신고(집합투자증권-신탁형)",
    "G002": "증권신고(집합투자증권-회사형)",
    "G003": "증권신고(집합투자증권-합병)",
    "H001": "자산유동화계획/양도등록",
    "H002": "사업/반기/분기보고서",
    "H003": "증권신고(유동화증권등)",
    "H004": "채권유동화계획/양도등록",
    "H005": "수시보고",
    "H006": "주요사항보고서",
    "I001": "수시공시",
    "I002": "공정공시",
    "I003": "시장조치/안내",
    "I004": "지분공시",
    "I005": "증권투자회사",
    "I006": "채권공시",
    "J001": "대규모내부거래관련",
    "J002": "대규모내부거래관련(구)",
    "J004": "기업집단현황공시",
    "J005": "비상장회사중요사항공시",
    "J006": "기타공정위공시",
}

OPENAI_DEFAULT_FAST_MODEL = "gpt-5.4-mini"
OPENAI_DEFAULT_REASONING_MODEL = "gpt-5.4"
GEMINI_DEFAULT_FAST_MODEL = "gemini-3.1-flash-lite-preview"
GEMINI_DEFAULT_REASONING_MODEL = "gemini-3.1-flash-lite-preview"
GEMINI_DEFAULT_FAST_THINKING_LEVEL = "minimal"
GEMINI_DEFAULT_REASONING_THINKING_LEVEL = "high"
OPENAI_DEFAULT_MODEL = OPENAI_DEFAULT_FAST_MODEL
GEMINI_DEFAULT_MODEL = GEMINI_DEFAULT_FAST_MODEL

RECENCY_MARKERS = (
    "최근", "최신", "가장 최근", "최근 연말", "최신 연말", "최근 공시", "최신 공시",
    "올해", "작년", "전년", "지난해", "이번 분기", "이번 반기", "최근 분기",
    "latest", "recent", "most recent", "this year", "last year", "current year",
    "this quarter", "latest filing", "recent filing",
)

FAILURE_TYPE_LABELS = {
    "company_identification_failed": "회사 식별 실패",
    "report_not_found": "보고서 없음",
    "section_not_found": "섹션 못 찾음",
    "insufficient_evidence": "근거 부족",
    "recent_filings_fallback_failed": "최신 공시 fallback 실패",
    "unknown_failure": "알 수 없는 실패",
}


class DartMaterialResult(TypedDict, total=False):
    """?? DART tool ??? / Public DART tool result."""

    ok: bool
    text: str
    source_paths: List[str]
    error: str


class OpenDartReportExplorer:
    CORP_CODE_CACHE_TTL = dt.timedelta(hours=24)
    _shared_corp_df: Optional[pd.DataFrame] = None
    _shared_corp_loaded_at: Optional[dt.datetime] = None
    """
    OpenDART API? DART viewer HTML? ?? ??? ?? ??, ?? ??, ?? ??? ?????.
    Uses the OpenDART API together with DART viewer HTML to search filings, traverse documents, and extract sections.

    ?? ??? ?? / Main data sources:
    - ????: corpCode.xml / Company codes: corpCode.xml
    - ????: list.json / Filing search: list.json
    - ?? ZIP ??: document.xml / Document ZIP listing: document.xml
    - ?? ??? ??: DART viewer(main.do / viewer.do) / Viewer page traversal: DART viewer(main.do / viewer.do)
    """

    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._corp_df: Optional[pd.DataFrame] = None
        self._company_overview_cache: Dict[str, Dict[str, Any]] = {}
        self._page_html_cache: Dict[str, str] = {}
        self._document_archive_cache: Dict[str, Dict[str, str]] = {}

    @classmethod
    def _corp_cache_dir(cls) -> str:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "output", "cache")

    @classmethod
    def _corp_cache_path(cls) -> str:
        return os.path.join(cls._corp_cache_dir(), "corp_codes.csv")

    @classmethod
    def _corp_cache_modified_at(cls) -> Optional[dt.datetime]:
        cache_path = cls._corp_cache_path()
        if not os.path.exists(cache_path):
            return None
        return dt.datetime.fromtimestamp(os.path.getmtime(cache_path))

    @classmethod
    def _is_cache_timestamp_fresh(cls, timestamp: Optional[dt.datetime]) -> bool:
        if timestamp is None:
            return False
        return (dt.datetime.now() - timestamp) < cls.CORP_CODE_CACHE_TTL

    @classmethod
    def _shared_corp_cache_fresh(cls) -> bool:
        return cls._shared_corp_df is not None and cls._is_cache_timestamp_fresh(cls._shared_corp_loaded_at)

    @classmethod
    def _load_corp_df_from_disk_cache(cls) -> Optional[pd.DataFrame]:
        cache_path = cls._corp_cache_path()
        if not os.path.exists(cache_path):
            return None

        modified_at = cls._corp_cache_modified_at()
        if not cls._is_cache_timestamp_fresh(modified_at):
            return None

        try:
            df = pd.read_csv(cache_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except Exception:
            return None

        if df.empty:
            return None

        required_columns = ["corp_code", "corp_name", "corp_eng_name", "stock_code", "modify_date"]
        for column in required_columns:
            if column not in df.columns:
                df[column] = ""
        return df[required_columns].copy()

    @classmethod
    def _save_corp_df_to_disk_cache(cls, df: pd.DataFrame) -> None:
        os.makedirs(cls._corp_cache_dir(), exist_ok=True)
        df.to_csv(cls._corp_cache_path(), index=False, encoding="utf-8-sig")

    @classmethod
    def _remember_corp_df(cls, df: pd.DataFrame, loaded_at: Optional[dt.datetime] = None) -> pd.DataFrame:
        copied = df.copy()
        cls._shared_corp_df = copied
        cls._shared_corp_loaded_at = loaded_at or dt.datetime.now()
        return copied

    def _get(self, url: str, **kwargs) -> requests.Response:
        last_error: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                resp = self.session.get(url, timeout=self.timeout, **kwargs)
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= 3:
                    break
                time.sleep(0.35 * attempt + random.uniform(0.05, 0.2))
        assert last_error is not None
        raise last_error

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").split())

    @staticmethod
    def _compact_text(text: str) -> str:
        return re.sub(r"\s+", "", text or "")

    @staticmethod
    def _decode_bytes(data: bytes) -> str:
        for enc in ("utf-8", "euc-kr", "cp949"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def _opendart_error_from_xml_bytes(content: bytes) -> Optional[str]:
        try:
            soup = BeautifulSoup(content, "xml")
            status_tag = soup.find("status")
            if not status_tag:
                return None
            status = status_tag.get_text(strip=True)
            message_tag = soup.find("message")
            message = message_tag.get_text(" ", strip=True) if message_tag else ""
            if status != "000":
                return f"OpenDART API error {status}: {message}"
        except Exception:
            return None
        return None

    # -------------------------------------------------
    # 1) 회사코드 / Company Codes
    # -------------------------------------------------
    def get_corp_codes(self, force_refresh: bool = False) -> pd.DataFrame:
        if self._corp_df is not None and not force_refresh:
            return self._corp_df.copy()

        if not force_refresh and self._shared_corp_cache_fresh():
            self._corp_df = self._shared_corp_df.copy()
            return self._corp_df.copy()

        if not force_refresh:
            cached_df = self._load_corp_df_from_disk_cache()
            if cached_df is not None:
                self._corp_df = self._remember_corp_df(
                    cached_df,
                    loaded_at=self._corp_cache_modified_at(),
                )
                return self._corp_df.copy()

        resp = self._get(
            f"{OPENDART_API}/corpCode.xml",
            params={"crtfc_key": self.api_key},
        )
        err = self._opendart_error_from_xml_bytes(resp.content)
        if err:
            raise RuntimeError(err)

        try:
            zf = zipfile.ZipFile(io.BytesIO(resp.content))
        except zipfile.BadZipFile as e:
            raise RuntimeError("corpCode.xml 응답이 ZIP 파일이 아닙니다.") from e

        xml_name = sorted(zf.namelist())[0]
        xml_text = self._decode_bytes(zf.read(xml_name))
        soup = BeautifulSoup(xml_text, "xml")

        rows: List[Dict[str, str]] = []
        for item in soup.find_all("list"):
            def _tag_text(tag_name: str) -> str:
                tag = item.find(tag_name)
                return tag.get_text(strip=True) if tag else ""

            rows.append(
                {
                    "corp_code": _tag_text("corp_code"),
                    "corp_name": _tag_text("corp_name"),
                    "corp_eng_name": _tag_text("corp_eng_name"),
                    "stock_code": _tag_text("stock_code"),
                    "modify_date": _tag_text("modify_date"),
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("회사코드 목록 파싱 결과가 비어 있습니다.")

        self._save_corp_df_to_disk_cache(df)
        self._corp_df = self._remember_corp_df(df)
        return self._corp_df.copy()

    def find_corp_exact(self, corp_name: str) -> Dict[str, str]:
        df = self.get_corp_codes()
        matched = df[df["corp_name"] == corp_name].copy()
        if matched.empty:
            raise ValueError(f"회사명을 정확히 찾지 못했습니다: {corp_name}")
        return matched.iloc[0].to_dict()

    def find_corp_by_stock_code(self, stock_code: str) -> Dict[str, str]:
        code = str(stock_code or "").strip()
        if not code:
            raise ValueError("stock_code is required")
        df = self.get_corp_codes()
        matched = df[df["stock_code"].astype(str).str.strip() == code].copy()
        if matched.empty:
            raise ValueError(f"종목코드에 대응하는 회사를 찾지 못했습니다: {stock_code}")
        return matched.iloc[0].to_dict()

    def get_company_overview(self, corp_code: str) -> Dict[str, Any]:
        corp_code_text = str(corp_code or "").strip()
        if not corp_code_text:
            raise ValueError("corp_code is required")

        cached = self._company_overview_cache.get(corp_code_text)
        if cached is not None:
            return dict(cached)

        resp = self._get(
            f"{OPENDART_API}/company.json",
            params={
                "crtfc_key": self.api_key,
                "corp_code": corp_code_text,
            },
        )
        data = resp.json()
        status = data.get("status")
        message = data.get("message", "")
        if status != "000":
            raise RuntimeError(f"OpenDART API error {status}: {message}")

        overview = dict(data)
        self._company_overview_cache[corp_code_text] = overview
        return dict(overview)

    # -------------------------------------------------
    # 2) 공시 검색 / Filing Search
    # -------------------------------------------------
    def search_filings(
        self,
        corp_code: str,
        bgn_de: str,
        end_de: str,
        pblntf_ty: Optional[str] = None,
        pblntf_detail_ty: Optional[str] = None,
        page_count: int = 100,
        last_reprt_at: str = "N",
        report_type: Optional[str] = None,
        report_name_query: Optional[str] = None,
    ) -> pd.DataFrame:
        all_rows: List[dict] = []
        page_no = 1

        while True:
            detail_filter = pblntf_detail_ty
            report_type_norm = (report_type or "").strip().lower()
            if detail_filter == "A001" and report_type_norm in {
                "all", "regular", "periodic", "semiannual", "half", "halfyear", "quarter", "quarterly"
            }:
                detail_filter = None

            params = {
                "crtfc_key": self.api_key,
                "corp_code": corp_code,
                "bgn_de": bgn_de,
                "end_de": end_de,
                "last_reprt_at": last_reprt_at,
                "sort": "date",
                "sort_mth": "desc",
                "page_no": page_no,
                "page_count": page_count,
            }
            if pblntf_ty:
                params["pblntf_ty"] = pblntf_ty
            if detail_filter:
                params["pblntf_detail_ty"] = detail_filter
            resp = self._get(f"{OPENDART_API}/list.json", params=params)
            data = resp.json()

            status = data.get("status")
            message = data.get("message", "")
            if status == "013":
                return pd.DataFrame()
            if status != "000":
                raise RuntimeError(f"OpenDART API error {status}: {message}")

            rows = data.get("list", [])
            all_rows.extend(rows)

            total_page = int(data.get("total_page", 1))
            if page_no >= total_page:
                break
            page_no += 1

        df = pd.DataFrame(all_rows)
        if df.empty:
            return df

        if report_type:
            mask = df["report_nm"].fillna("").apply(
                lambda report_nm: self._report_name_matches_type(report_nm, report_type)
            )
            df = df[mask].reset_index(drop=True)

        if report_name_query:
            query_norm = self._normalize_text(report_name_query).lower()
            mask = df["report_nm"].fillna("").apply(
                lambda report_nm: query_norm in self._normalize_text(report_nm).lower()
            )
            df = df[mask].reset_index(drop=True)

        return df

    @classmethod
    def _report_name_matches_type(cls, report_nm: str, report_type: str) -> bool:
        report_type_norm = (report_type or "").strip().lower()
        report_nm_norm = cls._normalize_text(report_nm)

        if report_type_norm in {"", "all"}:
            return True
        if report_type_norm in {"business", "annual"}:
            return report_nm_norm.startswith("사업보고서")
        if report_type_norm in {"semiannual", "half", "halfyear"}:
            return report_nm_norm.startswith("반기보고서")
        if report_type_norm in {"quarter", "quarterly"}:
            return report_nm_norm.startswith("분기보고서")
        if report_type_norm in {"regular", "periodic"}:
            return report_nm_norm.startswith(("사업보고서", "반기보고서", "분기보고서"))

        raise ValueError("report_type must be one of all/business/semiannual/quarterly/regular")

    # -------------------------------------------------
    # 3) 원문 ZIP 파일 목록 / Document ZIP Listing (document.xml)
    # -------------------------------------------------
    def fetch_document_archive(self, rcept_no: str) -> Dict[str, Any]:
        resp = self._get(
            f"{OPENDART_API}/document.xml",
            params={"crtfc_key": self.api_key, "rcept_no": rcept_no},
        )
        err = self._opendart_error_from_xml_bytes(resp.content)
        if err:
            raise RuntimeError(err)

        try:
            zf = zipfile.ZipFile(io.BytesIO(resp.content))
        except zipfile.BadZipFile as e:
            raise RuntimeError("document.xml 응답이 ZIP 파일이 아닙니다.") from e

        members = []
        for name in zf.namelist():
            members.append({"name": name, "size": zf.getinfo(name).file_size})
        return {"member_count": len(members), "members": members}

    def get_document_archive_texts(self, rcept_no: str) -> Dict[str, str]:
        if rcept_no in self._document_archive_cache:
            return dict(self._document_archive_cache[rcept_no])

        resp = self._get(
            f"{OPENDART_API}/document.xml",
            params={"crtfc_key": self.api_key, "rcept_no": rcept_no},
        )
        err = self._opendart_error_from_xml_bytes(resp.content)
        if err:
            raise RuntimeError(err)

        try:
            zf = zipfile.ZipFile(io.BytesIO(resp.content))
        except zipfile.BadZipFile as e:
            raise RuntimeError("document.xml 응답이 ZIP 파일이 아닙니다.") from e

        texts: Dict[str, str] = {}
        for name in zf.namelist():
            texts[name] = self._decode_bytes(zf.read(name))
        self._document_archive_cache[rcept_no] = texts
        return dict(texts)

    @staticmethod
    def _extract_xml_text_label(fragment: str) -> str:
        label = re.sub(r"<[^>]+>", " ", fragment)
        label = BeautifulSoup(label, "html.parser").get_text(" ", strip=True)
        return OpenDartReportExplorer._normalize_text(label)

    @staticmethod
    def _iter_xml_toc_matches(xml_text: str) -> List[Dict[str, Any]]:
        pattern = re.compile(
            r'<(?P<tag>[A-Z][A-Z0-9\-]*)(?P<attrs>[^>]*?\bATOC="Y"[^>]*?\bATOCID="(?P<atocid>\d+)"[^>]*)>(?P<inner>.*?)</(?P=tag)>',
            re.DOTALL,
        )
        matches: List[Dict[str, Any]] = []
        for match in pattern.finditer(xml_text):
            label = OpenDartReportExplorer._extract_xml_text_label(match.group("inner"))
            if not label:
                continue
            matches.append(
                {
                    "tag": match.group("tag"),
                    "atocid": match.group("atocid"),
                    "title": label,
                    "start": match.start(),
                    "end": match.end(),
                }
            )
        return matches

    @staticmethod
    def _xml_heading_level(title: str) -> int:
        text = str(title or "").strip()
        compact = text.replace(" ", "")
        if re.match(r"^[IVX]+\.", text):
            return 1
        if text.startswith("【") and text.endswith("】"):
            return 1
        if re.match(r"^\d+\-\d+\.", text):
            return 3
        if re.match(r"^\d+\.", text):
            return 2
        if re.match(r"^[가-힣]\.", text):
            return 4
        if re.match(r"^\(\d+\)", compact):
            return 5
        return 1

    def list_document_toc_entries(self, rcept_no: str) -> List[Dict[str, Any]]:
        archive_texts = self.get_document_archive_texts(rcept_no)
        toc_entries: List[Dict[str, Any]] = []
        order = 1
        for member_name, xml_text in archive_texts.items():
            if not member_name.lower().endswith(".xml"):
                continue
            matches = self._iter_xml_toc_matches(xml_text)
            if not matches:
                continue

            hierarchy: Dict[int, str] = {}
            doc_title = matches[0]["title"]
            for match in matches:
                title = match["title"]
                level = self._xml_heading_level(title)
                hierarchy[level] = title
                for key in list(hierarchy.keys()):
                    if key > level:
                        hierarchy.pop(key, None)
                parent_title = hierarchy.get(level - 1) if level > 1 else None
                toc_entries.append(
                    {
                        "order": order,
                        "xml_member": member_name,
                        "doc_title": doc_title,
                        "section_title": title,
                        "toc_id": match["atocid"],
                        "tag": match["tag"],
                        "level": level,
                        "parent_title": parent_title,
                    }
                )
                order += 1
        return toc_entries

    @staticmethod
    def _xml_fragment_to_blocks(fragment: str, max_blocks: int = 60) -> List[Dict[str, Any]]:
        normalized = fragment or ""
        normalized = re.sub(r"</(TD|TH)>", " | ", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"</TR>", "\n", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"</(P|TITLE|COVER-TITLE|SUBTITLE|LI|TABLE|TBODY|THEAD|SECTION)>", "\n", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"<br\s*/?>", "\n", normalized, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", normalized)
        text = BeautifulSoup(text, "html.parser").get_text("\n", strip=True)
        lines = [
            OpenDartReportExplorer._normalize_text(line)
            for line in text.splitlines()
        ]
        lines = [line for line in lines if line]
        if not lines:
            return []
        blocks: List[Dict[str, Any]] = []
        for line in lines[:max_blocks]:
            blocks.append({"type": "text", "content": line[:10000]})
        return blocks

    def extract_document_toc_section(
        self,
        rcept_no: str,
        xml_member: str,
        toc_id: str,
        max_blocks: int = 80,
    ) -> Dict[str, Any]:
        archive_texts = self.get_document_archive_texts(rcept_no)
        if xml_member not in archive_texts:
            raise ValueError(f"unknown xml_member: {xml_member}")
        xml_text = archive_texts[xml_member]
        matches = self._iter_xml_toc_matches(xml_text)
        target_index = None
        for index, match in enumerate(matches):
            if str(match["atocid"]) == str(toc_id):
                target_index = index
                break
        if target_index is None:
            raise ValueError(f"toc_id not found in xml member: {toc_id}")

        current = matches[target_index]
        next_start = matches[target_index + 1]["start"] if target_index + 1 < len(matches) else len(xml_text)
        fragment = xml_text[current["start"]:next_start]
        blocks = self._xml_fragment_to_blocks(fragment, max_blocks=max_blocks)
        return {
            "mode": "xml_atoc",
            "matched_section_title": current["title"],
            "matched_toc_id": str(current["atocid"]),
            "available_section_count": len(matches),
            "blocks": blocks,
        }

    # -------------------------------------------------
    # 4) DART viewer 기준 하위문서 / 첨부문서 탐색 / Viewer Subdocument Traversal
    # -------------------------------------------------
    def get_sub_docs(self, rcp_no_or_url: str) -> pd.DataFrame:
        if rcp_no_or_url.isdecimal():
            resp = self._get(f"{DART_WEB}/dsaf001/main.do", params={"rcpNo": rcp_no_or_url})
        elif rcp_no_or_url.startswith("http"):
            resp = self._get(rcp_no_or_url)
        else:
            raise ValueError("invalid `rcp_no_or_url`")

        html = resp.text

        multi_page_re = (
            r"\s+node[12]\['text'\][ =]+\"(.*?)\"\;"
            r"\s+node[12]\['id'\][ =]+\"(\d+)\"\;"
            r"\s+node[12]\['rcpNo'\][ =]+\"(\d+)\"\;"
            r"\s+node[12]\['dcmNo'\][ =]+\"(\d+)\"\;"
            r"\s+node[12]\['eleId'\][ =]+\"(\d+)\"\;"
            r"\s+node[12]\['offset'\][ =]+\"(\d+)\"\;"
            r"\s+node[12]\['length'\][ =]+\"(\d+)\"\;"
            r"\s+node[12]\['dtd'\][ =]+\"(.*?)\"\;"
            r"\s+node[12]\['tocNo'\][ =]+\"(\d+)\"\;"
        )
        matches = re.findall(multi_page_re, html)

        rows: List[Dict[str, str]] = []
        if matches:
            for m in matches:
                raw_title = m[0]
                title = BeautifulSoup(raw_title, "html.parser").get_text(" ", strip=True)
                title = self._normalize_text(title)
                params = {
                    "rcpNo": m[2],
                    "dcmNo": m[3],
                    "eleId": m[4],
                    "offset": m[5],
                    "length": m[6],
                    "dtd": m[7],
                }
                url = f"{DART_WEB}/report/viewer.do?{urlencode(params)}"
                rows.append({"title": title, "url": url})
        else:
            single_page_re = r"viewDoc\('(\d+)', '(\d+)', '(\d+)', '(\d+)', '(\d+)', '(\S+)',''\)\;"
            matches2 = re.findall(single_page_re, html)
            if matches2:
                m = matches2[0]
                title = BeautifulSoup(html, "lxml").title.get_text(strip=True)
                params = {
                    "rcpNo": m[0],
                    "dcmNo": m[1],
                    "eleId": m[2],
                    "offset": m[3],
                    "length": m[4],
                    "dtd": m[5],
                }
                url = f"{DART_WEB}/report/viewer.do?{urlencode(params)}"
                rows.append({"title": title, "url": url})

        if not rows:
            return pd.DataFrame(columns=["title", "url"])

        return pd.DataFrame(rows).drop_duplicates(subset=["url"]).reset_index(drop=True)

    def get_attach_docs(self, rcp_no: str) -> pd.DataFrame:
        resp = self._get(f"{DART_WEB}/dsaf001/main.do", params={"rcpNo": rcp_no})
        soup = BeautifulSoup(resp.text, "lxml")

        att = soup.find(id="att")
        if not att:
            return pd.DataFrame(columns=["title", "url"])

        rows = []
        for opt in att.find_all("option"):
            value = opt.get("value")
            if not value or value == "null":
                continue
            title = self._normalize_text(opt.get_text(" ", strip=True))
            url = f"{DART_WEB}/dsaf001/main.do?{value}"
            rows.append({"title": title, "url": url})

        if not rows:
            return pd.DataFrame(columns=["title", "url"])

        return pd.DataFrame(rows).drop_duplicates(subset=["url"]).reset_index(drop=True)

    def infer_page_kind(self, title: str) -> str:
        compact = self._compact_text(title)

        if "연결재무제표주석" in compact:
            return "note_consolidated"
        if "재무제표주석" in compact and "연결" not in compact:
            return "note_separate"

        financial_markers = (
            "재무상태표", "손익계산서", "포괄손익계산서", "현금흐름표", "자본변동표"
        )
        if "연결" in compact and any(marker in compact for marker in financial_markers):
            return "financial_statement_consolidated"
        if any(marker in compact for marker in financial_markers):
            return "financial_statement_separate"

        if "연결감사보고서" in compact:
            return "audit_consolidated"
        if "감사보고서" in compact:
            return "audit"
        if "사업의내용" in compact:
            return "business"
        if "위험요소" in compact or "리스크" in compact:
            return "risk"
        if "이사의경영진단및분석의견" in compact:
            return "mdna"
        if "주요계약" in compact:
            return "contracts"
        return "other"

    def get_all_report_pages(self, rcp_no: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        main_sub = self.get_sub_docs(rcp_no)
        for _, row in main_sub.iterrows():
            rows.append(
                {
                    "source": "main_sub",
                    "parent_title": None,
                    "title": row["title"],
                    "url": row["url"],
                }
            )

        attach_docs = self.get_attach_docs(rcp_no)
        for _, att in attach_docs.iterrows():
            att_title = att["title"]
            att_url = att["url"]

            rows.append(
                {
                    "source": "attach_main",
                    "parent_title": att_title,
                    "title": att_title,
                    "url": att_url,
                }
            )

            sub_df = self.get_sub_docs(att_url)
            for _, row in sub_df.iterrows():
                rows.append(
                    {
                        "source": "attach_sub",
                        "parent_title": att_title,
                        "title": row["title"],
                        "url": row["url"],
                    }
                )

        if not rows:
            return []

        df = pd.DataFrame(rows).drop_duplicates(subset=["url"]).reset_index(drop=True)
        items = df.to_dict(orient="records")
        for item in items:
            item["kind"] = self.infer_page_kind(item["title"])
        return items

    # -------------------------------------------------
    # 5) 페이지 HTML / 목차 / 섹션 추출 / Page HTML, TOC, and Section Extraction
    # -------------------------------------------------
    def get_page_html(self, url: str) -> str:
        if url in self._page_html_cache:
            return self._page_html_cache[url]
        resp = self._get(url)
        self._page_html_cache[url] = resp.text
        return resp.text

    def get_page_soup(self, url: str) -> BeautifulSoup:
        return BeautifulSoup(self.get_page_html(url), "lxml")

    def list_page_subsections(self, url: str, page_title: str) -> List[Dict[str, str]]:
        soup = self.get_page_soup(url)
        toc_tags = soup.find_all(
            "a",
            attrs={"name": lambda x: x and str(x).startswith("toc")}
        )

        seen = set()
        sections: List[Dict[str, str]] = []
        title_norm = self._normalize_text(page_title)

        for tag in toc_tags:
            parent = tag.parent if tag.parent else tag
            section_title = self._normalize_text(parent.get_text(" ", strip=True))
            toc_id = tag.get("name", "").strip()

            if not section_title or section_title == title_norm:
                continue

            key = (section_title, toc_id)
            if key in seen:
                continue
            seen.add(key)

            sections.append({"title": section_title, "toc_id": toc_id})

        return sections

    @staticmethod
    def _json_safe_value(v: Any) -> Any:
        if pd.isna(v):
            return None
        if isinstance(v, pd.Timestamp):
            return v.isoformat()
        return v

    def _serialize_df(self, df: pd.DataFrame, max_rows: int) -> Dict[str, Any]:
        df2 = df.copy().head(max_rows)
        return {
            "columns": [str(c) for c in df2.columns.tolist()],
            "rows": [
                [self._json_safe_value(v) for v in row]
                for row in df2.itertuples(index=False, name=None)
            ],
            "row_count_returned": len(df2),
            "row_count_original": len(df),
        }

    def _extract_blocks_between(
        self,
        soup: BeautifulSoup,
        start_toc_id: str,
        end_toc_id: Optional[str],
        max_table_rows: int,
        max_blocks: int,
    ) -> List[Dict[str, Any]]:
        start_anchor = soup.find("a", attrs={"name": start_toc_id})
        if not start_anchor:
            raise ValueError("HTML 내에서 해당 목차 시작점을 찾지 못했습니다.")

        current_node = start_anchor.parent if start_anchor.parent else start_anchor
        blocks: List[Dict[str, Any]] = []

        for sibling in current_node.find_next_siblings():
            if end_toc_id and sibling.find("a", attrs={"name": end_toc_id}):
                break

            if len(blocks) >= max_blocks:
                break

            if sibling.name in {"p", "div"}:
                text = self._normalize_text(sibling.get_text(" ", strip=True))
                if text:
                    blocks.append({"type": "text", "content": text})
            elif sibling.name == "table":
                classes = sibling.get("class", []) or []
                if "nb" in classes:
                    text = self._normalize_text(sibling.get_text(" | ", strip=True))
                    if text:
                        blocks.append({"type": "text", "content": text})
                else:
                    try:
                        df_list = pd.read_html(io.StringIO(str(sibling)))
                        if df_list and not df_list[0].empty:
                            df = df_list[0].dropna(how="all", axis=0).dropna(how="all", axis=1)
                            blocks.append(
                                {
                                    "type": "table",
                                    "content": self._serialize_df(df, max_rows=max_table_rows),
                                }
                            )
                    except Exception:
                        pass

        return blocks

    def _extract_whole_page_blocks(
        self,
        soup: BeautifulSoup,
        max_table_rows: int,
        max_blocks: int,
    ) -> List[Dict[str, Any]]:
        body = soup.body or soup
        blocks: List[Dict[str, Any]] = []

        for tag in body.find_all(["p", "table"], recursive=True):
            if len(blocks) >= max_blocks:
                break

            if tag.name == "p":
                if tag.find_parent("table"):
                    continue
                text = self._normalize_text(tag.get_text(" ", strip=True))
                if text:
                    blocks.append({"type": "text", "content": text})
            elif tag.name == "table":
                classes = tag.get("class", []) or []
                if "nb" in classes:
                    text = self._normalize_text(tag.get_text(" | ", strip=True))
                    if text:
                        blocks.append({"type": "text", "content": text})
                else:
                    try:
                        df_list = pd.read_html(io.StringIO(str(tag)))
                        if df_list and not df_list[0].empty:
                            df = df_list[0].dropna(how="all", axis=0).dropna(how="all", axis=1)
                            blocks.append(
                                {
                                    "type": "table",
                                    "content": self._serialize_df(df, max_rows=max_table_rows),
                                }
                            )
                    except Exception:
                        pass

        if not blocks:
            text = self._normalize_text(body.get_text(" ", strip=True))
            if text:
                blocks.append({"type": "text", "content": text[:10000]})

        return blocks

    def extract_report_section(
        self,
        url: str,
        page_title: str,
        keyword: Optional[str] = None,
        toc_id: Optional[str] = None,
        whole_page: bool = False,
        max_table_rows: int = 20,
        max_blocks: int = 80,
    ) -> Dict[str, Any]:
        soup = self.get_page_soup(url)
        sections = self.list_page_subsections(url, page_title)

        if whole_page or (not keyword and not toc_id):
            blocks = self._extract_whole_page_blocks(
                soup=soup,
                max_table_rows=max_table_rows,
                max_blocks=max_blocks,
            )
            return {
                "mode": "whole_page",
                "matched_section_title": None,
                "matched_toc_id": None,
                "available_section_count": len(sections),
                "blocks": blocks,
            }

        matched_title = None
        start_toc_id = None
        end_toc_id = None

        if toc_id:
            for i, sec in enumerate(sections):
                if sec["toc_id"] == toc_id:
                    start_toc_id = sec["toc_id"]
                    matched_title = sec["title"]
                    if i + 1 < len(sections):
                        end_toc_id = sections[i + 1]["toc_id"]
                    break
            if not start_toc_id:
                raise ValueError(f"toc_id를 찾지 못했습니다: {toc_id}")
        else:
            for i, sec in enumerate(sections):
                if keyword and keyword in sec["title"]:
                    start_toc_id = sec["toc_id"]
                    matched_title = sec["title"]
                    if i + 1 < len(sections):
                        end_toc_id = sections[i + 1]["toc_id"]
                    break
            if not start_toc_id:
                raise ValueError(f"keyword를 포함하는 목차를 찾지 못했습니다: {keyword}")

        blocks = self._extract_blocks_between(
            soup=soup,
            start_toc_id=start_toc_id,
            end_toc_id=end_toc_id,
            max_table_rows=max_table_rows,
            max_blocks=max_blocks,
        )
        return {
            "mode": "section",
            "matched_section_title": matched_title,
            "matched_toc_id": start_toc_id,
            "available_section_count": len(sections),
            "blocks": blocks,
        }

    @staticmethod
    def _format_amount_for_markdown(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if not text or text == "-":
            return text

        negative = text.startswith("-")
        plain = text.replace(",", "")
        if plain.startswith("(") and plain.endswith(")"):
            negative = True
            plain = plain[1:-1]
        if plain.isdigit():
            formatted = f"{int(plain):,}"
            return f"-{formatted}" if negative else formatted
        return text

    @staticmethod
    def _markdown_escape(text: Any) -> str:
        return str(text).replace("|", "\\|")

    def _render_serialized_table_markdown(self, table: Dict[str, Any]) -> str:
        columns = [self._markdown_escape(col) for col in table.get("columns", [])]
        rows = table.get("rows", [])
        if not columns:
            return ""

        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for row in rows:
            rendered = [
                self._markdown_escape(self._format_amount_for_markdown(cell))
                for cell in row
            ]
            if len(rendered) < len(columns):
                rendered.extend([""] * (len(columns) - len(rendered)))
            lines.append("| " + " | ".join(rendered[: len(columns)]) + " |")
        return "\n".join(lines)

    def _render_blocks_as_markdown(self, blocks: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for block in blocks:
            if block["type"] == "text":
                parts.append(block["content"])
            elif block["type"] == "table":
                table_md = self._render_serialized_table_markdown(block["content"])
                if table_md:
                    parts.append(table_md)
        return "\n\n".join(part for part in parts if part)

    def _find_filing_by_rcept_no(self, corp_code: str, rcept_no: str) -> Dict[str, Any]:
        filing_year = rcept_no[:4]
        filings = self.search_filings(
            corp_code=corp_code,
            bgn_de=f"{filing_year}0101",
            end_de=f"{filing_year}1231",
            pblntf_detail_ty=None,
            last_reprt_at="N",
            report_type="regular",
        )
        if filings.empty:
            raise ValueError(f"rcept_no에 해당하는 정기보고서를 찾지 못했습니다: {rcept_no}")

        matched = filings[filings["rcept_no"] == rcept_no]
        if matched.empty:
            raise ValueError(f"corp_code/rcept_no 조합에 해당하는 정기보고서를 찾지 못했습니다: {rcept_no}")
        return matched.iloc[0].to_dict()

    @staticmethod
    def _extract_business_year(report_nm: str) -> str:
        matched = re.search(r"\((\d{4})\.\d{2}\)", report_nm or "")
        if not matched:
            raise ValueError(f"report_nm에서 사업연도를 파싱하지 못했습니다: {report_nm}")
        return matched.group(1)

    @staticmethod
    def _reprt_code_from_report_nm(report_nm: str) -> str:
        report_nm = report_nm or ""
        if report_nm.startswith("사업보고서"):
            return "11011"
        if report_nm.startswith("반기보고서"):
            return "11012"
        if report_nm.startswith("분기보고서"):
            month_match = re.search(r"\(\d{4}\.(\d{2})\)", report_nm)
            if month_match and month_match.group(1) == "09":
                return "11014"
            return "11013"
        raise ValueError(f"정기보고서 유형을 판별하지 못했습니다: {report_nm}")

    @classmethod
    def try_extract_business_year(cls, report_nm: str) -> Optional[str]:
        try:
            return cls._extract_business_year(report_nm)
        except Exception:
            return None

    @classmethod
    def try_reprt_code_from_report_nm(cls, report_nm: str) -> Optional[str]:
        try:
            return cls._reprt_code_from_report_nm(report_nm)
        except Exception:
            return None

    @staticmethod
    def _fs_div_from_scope(statement_scope: str) -> tuple[str, str]:
        scope_norm = (statement_scope or "").strip().lower()
        if scope_norm in {"consolidated", "cfs", "연결"}:
            return "CFS", "연결"
        if scope_norm in {"separate", "ofs", "individual", "별도", "일반"}:
            return "OFS", "별도"
        raise ValueError("statement_scope must be 'consolidated' or 'separate'")

    def _fetch_financial_statement_rows(
        self,
        corp_code: str,
        bsns_year: str,
        reprt_code: str,
        fs_div: str,
    ) -> pd.DataFrame:
        resp = self._get(
            f"{OPENDART_API}/fnlttSinglAcntAll.json",
            params={
                "crtfc_key": self.api_key,
                "corp_code": corp_code,
                "bsns_year": bsns_year,
                "reprt_code": reprt_code,
                "fs_div": fs_div,
            },
        )
        data = resp.json()
        status = data.get("status")
        message = data.get("message", "")
        if status != "000":
            raise RuntimeError(f"OpenDART API error {status}: {message}")

        df = pd.DataFrame(data.get("list", []))
        if df.empty:
            raise ValueError("재무제표 API 결과가 비어 있습니다.")
        return df

    def _render_statement_rows_markdown(self, rows: pd.DataFrame) -> str:
        if rows.empty:
            return "표시할 재무제표 행이 없습니다."

        df = rows.copy()
        df["ord_num"] = pd.to_numeric(df["ord"], errors="coerce").fillna(999999)
        df = df.sort_values(["sj_div", "ord_num", "account_nm"]).reset_index(drop=True)

        section_order = ["BS", "IS", "CIS", "SCE", "CF"]
        grouped: List[tuple[str, pd.DataFrame]] = []
        for sj_div in section_order:
            sub = df[df["sj_div"] == sj_div].copy()
            if not sub.empty:
                grouped.append((sub.iloc[0]["sj_nm"], sub))
        for sj_nm, sub in df.groupby("sj_nm", sort=False):
            if sub.iloc[0]["sj_div"] not in section_order:
                grouped.append((sj_nm, sub.copy()))

        parts: List[str] = []
        for sj_nm, sub in grouped:
            period_labels = [sub.iloc[0].get("thstrm_nm", "당기"), sub.iloc[0].get("frmtrm_nm", "전기")]
            include_bfefrm = bool(sub.iloc[0].get("bfefrmtrm_nm"))
            if include_bfefrm:
                period_labels.append(sub.iloc[0].get("bfefrmtrm_nm", "전전기"))

            table_lines = [
                f"### {sj_nm}",
                "| 계정 | " + " | ".join(self._markdown_escape(label) for label in period_labels) + " |",
                "| --- | " + " | ".join(["---"] * len(period_labels)) + " |",
            ]
            for _, row in sub.iterrows():
                cells = [
                    self._markdown_escape(row.get("account_nm", "")),
                    self._markdown_escape(self._format_amount_for_markdown(row.get("thstrm_amount", ""))),
                    self._markdown_escape(self._format_amount_for_markdown(row.get("frmtrm_amount", ""))),
                ]
                if include_bfefrm:
                    cells.append(
                        self._markdown_escape(
                            self._format_amount_for_markdown(row.get("bfefrmtrm_amount", ""))
                        )
                    )
                table_lines.append("| " + " | ".join(cells) + " |")
            parts.append("\n".join(table_lines))

        return "\n\n".join(parts)

    def _match_section_from_toc(
        self,
        sections: List[Dict[str, str]],
        keywords: List[str],
    ) -> Optional[Dict[str, str]]:
        best_section = None
        best_score = -1
        for section in sections:
            title_compact = self._compact_text(section["title"])
            for keyword in keywords:
                keyword_compact = self._compact_text(keyword)
                score = -1
                if title_compact == keyword_compact:
                    score = 200 + len(keyword_compact)
                elif keyword_compact and keyword_compact in title_compact:
                    score = 100 + len(keyword_compact)
                if score > best_score:
                    best_score = score
                    best_section = section
        return best_section

    @staticmethod
    def _looks_like_major_heading(text: str) -> bool:
        return bool(re.match(r"^\d+\.\s*", (text or "").strip()))

    def _extract_blocks_by_heading_keywords(
        self,
        url: str,
        keywords: List[str],
        max_table_rows: int,
        max_blocks: int,
    ) -> List[Dict[str, Any]]:
        soup = self.get_page_soup(url)
        blocks = self._extract_whole_page_blocks(
            soup=soup,
            max_table_rows=max_table_rows,
            max_blocks=max(max_blocks * 10, 300),
        )
        best_idx = None
        best_score = -1
        for idx, block in enumerate(blocks):
            if block["type"] != "text":
                continue
            compact_text = self._compact_text(block["content"])
            for keyword in keywords:
                keyword_compact = self._compact_text(keyword)
                score = -1
                if compact_text == keyword_compact:
                    score = 200 + len(keyword_compact)
                elif compact_text.startswith(keyword_compact):
                    score = 150 + len(keyword_compact)
                elif keyword_compact and keyword_compact in compact_text:
                    score = 100 + len(keyword_compact)
                if score > best_score:
                    best_score = score
                    best_idx = idx

        if best_idx is None:
            return []

        end_idx = len(blocks)
        for idx in range(best_idx + 1, len(blocks)):
            block = blocks[idx]
            if block["type"] == "text" and self._looks_like_major_heading(block["content"]):
                end_idx = idx
                break
        return blocks[best_idx: min(end_idx, best_idx + max_blocks)]

    def _extract_note_topic_markdown(
        self,
        page: Dict[str, Any],
        sections: List[Dict[str, str]],
        title: str,
        keywords: List[str],
        max_table_rows: int,
        max_blocks: int,
    ) -> str:
        matched = self._match_section_from_toc(sections, keywords)
        if matched:
            extracted = self.extract_report_section(
                url=page["url"],
                page_title=page["title"],
                toc_id=matched["toc_id"],
                whole_page=False,
                max_table_rows=max_table_rows,
                max_blocks=max_blocks,
            )
            body = self._render_blocks_as_markdown(extracted["blocks"])
            if body:
                return f"### {title}\n\n원문 섹션: {matched['title']}\n\n{body}"

        blocks = self._extract_blocks_by_heading_keywords(
            url=page["url"],
            keywords=keywords,
            max_table_rows=max_table_rows,
            max_blocks=max_blocks,
        )
        if blocks:
            body = self._render_blocks_as_markdown(blocks)
            if body:
                return f"### {title}\n\n{body}"

        return f"### {title}\n\n해당 주석 섹션을 찾지 못했습니다."

    def get_financial_statement_key_contents(
        self,
        corp_code: str,
        rcept_no: str,
        statement_scope: str = "consolidated",
        max_table_rows: int = 20,
        max_note_blocks: int = 30,
    ) -> str:
        filing = self._find_filing_by_rcept_no(corp_code=corp_code, rcept_no=rcept_no)
        report_nm = filing.get("report_nm", "")
        bsns_year = self._extract_business_year(report_nm)
        reprt_code = self._reprt_code_from_report_nm(report_nm)
        fs_div, scope_label = self._fs_div_from_scope(statement_scope)

        statement_rows = self._fetch_financial_statement_rows(
            corp_code=corp_code,
            bsns_year=bsns_year,
            reprt_code=reprt_code,
            fs_div=fs_div,
        )

        pages = self.get_all_report_pages(rcept_no)
        note_kind = "note_consolidated" if fs_div == "CFS" else "note_separate"
        note_candidates = [page for page in pages if page.get("kind") == note_kind]
        note_page = note_candidates[0] if note_candidates else None
        sections = self.list_page_subsections(note_page["url"], note_page["title"]) if note_page else []

        note_topics = [
            ("매출채권", ["매출채권", "매출채권 및 기타채권", "매출채권및기타채권", "매출채권과 계약자산"]),
            ("재고자산", ["재고자산"]),
            ("기타유동자산", ["기타유동자산", "기타자산", "기타유동자산 및 기타비유동자산", "선급금", "선급비용"]),
            ("무형자산", ["무형자산", "무형자산(영업권 제외)", "무형자산(영업권제외)"]),
            ("기타지급채무", ["매입채무 및 기타채무", "매입채무및기타채무", "기타채무", "기타지급채무", "매입채무"]),
            ("차입금", ["차입금 및 사채", "차입금", "사채", "차입부채"]),
            ("기타유동부채 및 기타비유동부채", ["기타유동부채", "기타비유동부채", "기타부채", "기타유동부채 및 기타비유동부채"]),
            ("판매비와관리비", ["판매비와관리비", "판매비 및 관리비", "판매관리비"]),
            ("비용의 성격별 분류", ["비용의 성격별 분류", "비용의 성격별분류", "비용의 성격", "성격별 비용"]),
            ("영업비용", ["영업비용", "영업비용의 성격별 분류", "영업비용의 성격", "매출원가", "매출원가 및 판매비와관리비"]),
            ("현금흐름표", ["현금흐름표", "현금흐름표에 관한 정보", "연결현금흐름표에 관한 정보", "별도현금흐름표에 관한 정보"]),
        ]

        statement_md = self._render_statement_rows_markdown(statement_rows)
        notes_md_parts: List[str] = []
        if note_page:
            for title, keywords in note_topics:
                notes_md_parts.append(
                    self._extract_note_topic_markdown(
                        page=note_page,
                        sections=sections,
                        title=title,
                        keywords=keywords,
                        max_table_rows=max_table_rows,
                        max_blocks=max_note_blocks,
                    )
                )
        else:
            notes_md_parts.append("주석 페이지를 찾지 못했습니다.")

        header = [
            "# 재무제표 주요내용",
            f"- corp_code: `{corp_code}`",
            f"- rcept_no: `{rcept_no}`",
            f"- 보고서명: {report_nm}",
            f"- 재무제표 구분: {scope_label}",
        ]
        return "\n".join(header) + "\n\n## 재무제표\n\n" + statement_md + "\n\n## 주요주석\n\n" + "\n\n".join(notes_md_parts)


class DartReportToolService:
    """
    OpenAI? Gemini function calling? ???? DART ??? ??????.
    Service layer for DART tools used by OpenAI and Gemini function calling.

    ?? ??, ??? ??, ???/?? ??? ?? ?? ??? ?????.
    Provides thin tool functions for company search, filing search, and page/section extraction.
    """

    def __init__(self, dart_api_key: str):
        self.explorer = OpenDartReportExplorer(api_key=dart_api_key)
        self._reports: Dict[str, Dict[str, Any]] = {}
        self._pages: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _report_window(report_year: int) -> tuple[str, str]:
        next_year = report_year + 1
        return f"{next_year}0101", f"{next_year}0430"

    @staticmethod
    def _make_id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def _require_report(self, report_id: str) -> Dict[str, Any]:
        if report_id not in self._reports:
            raise ValueError(f"unknown report_id: {report_id}")
        return self._reports[report_id]

    def _require_page(self, page_id: str) -> Dict[str, Any]:
        if page_id not in self._pages:
            raise ValueError(f"unknown page_id: {page_id}")
        return self._pages[page_id]

    def _build_report_payload(
        self,
        corp: Dict[str, Any],
        report_row: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        report_nm = report_row.get("report_nm")
        reprt_code = self.explorer.try_reprt_code_from_report_nm(report_nm)
        payload = {
            "report_id": self._make_id("report"),
            "corp_name": corp["corp_name"],
            "corp_code": corp["corp_code"],
            "stock_code": corp.get("stock_code"),
            "report_nm": report_nm,
            "rcept_no": report_row.get("rcept_no"),
            "rcept_dt": report_row.get("rcept_dt"),
            "business_year": self.explorer.try_extract_business_year(report_nm),
            "reprt_code": reprt_code,
            "reprt_code_label": REGULAR_REPRT_CODE_LABELS.get(reprt_code),
            "pblntf_ty": report_row.get("pblntf_ty"),
            "pblntf_detail_ty": report_row.get("pblntf_detail_ty"),
            "pblntf_detail_ty_label": PBLNTF_DETAIL_TYPE_LABELS.get(report_row.get("pblntf_detail_ty")),
        }
        if extra:
            payload.update(extra)
        self._reports[payload["report_id"]] = payload
        return payload

    @staticmethod
    def _pick_note_scope(kind: str, note_scope: str) -> bool:
        if note_scope == "consolidated":
            return kind == "note_consolidated"
        if note_scope == "separate":
            return kind == "note_separate"
        raise ValueError("note_scope must be 'consolidated' or 'separate'")

    @staticmethod
    def _infer_pblntf_ty(pblntf_detail_ty: Optional[str]) -> Optional[str]:
        detail = str(pblntf_detail_ty or "").strip().upper()
        if not detail:
            return None
        family = detail[:1]
        return family if family in PBLNTF_TYPE_LABELS else None

    @classmethod
    def _default_last_reprt_at(cls, pblntf_detail_ty: Optional[str], pblntf_ty: Optional[str]) -> str:
        family = str(pblntf_ty or cls._infer_pblntf_ty(pblntf_detail_ty) or "").strip().upper()
        return "Y" if family == "A" else "N"

    @classmethod
    def _normalize_report_name_query(
        cls,
        pblntf_detail_ty: Optional[str],
        pblntf_ty: Optional[str],
        report_name_query: Optional[str],
    ) -> Optional[str]:
        query = str(report_name_query or "").strip() or None
        if not query:
            return None
        return query

    @staticmethod
    def _normalize_search_text(text: Any) -> str:
        return OpenDartReportExplorer._normalize_text(str(text or "")).lower()

    @classmethod
    def _search_tokens(cls, text: Any) -> List[str]:
        normalized = cls._normalize_search_text(text)
        return [token for token in re.split(r"[^0-9a-zA-Z가-힣]+", normalized) if token]

    @classmethod
    def _score_query_match(cls, query: str, candidate: Any) -> int:
        query_norm = cls._normalize_search_text(query)
        candidate_norm = cls._normalize_search_text(candidate)
        if not query_norm or not candidate_norm:
            return 0

        query_compact = re.sub(r"\s+", "", query_norm)
        candidate_compact = re.sub(r"\s+", "", candidate_norm)

        score = 0
        if candidate_norm == query_norm:
            score += 1000
        elif candidate_compact == query_compact:
            score += 900
        elif query_norm in candidate_norm:
            score += 700 + min(len(query_norm), 40)
        elif query_compact and query_compact in candidate_compact:
            score += 650 + min(len(query_compact), 40)

        query_tokens = cls._search_tokens(query_norm)
        candidate_tokens = cls._search_tokens(candidate_norm)
        candidate_token_set = set(candidate_tokens)
        overlap = [token for token in query_tokens if token in candidate_token_set]
        if overlap:
            score += 80 * len(set(overlap))
            score += sum(min(len(token), 12) for token in set(overlap))

        return score

    def _store_page(self, report_id: str, page: Dict[str, Any]) -> Dict[str, Any]:
        for stored in self._pages.values():
            if stored["report_id"] == report_id and stored["url"] == page["url"]:
                return stored

        page_id = self._make_id("page")
        stored = {
            "page_id": page_id,
            "report_id": report_id,
            "title": page["title"],
            "url": page["url"],
            "source": page["source"],
            "parent_title": page.get("parent_title"),
            "kind": page.get("kind", "other"),
        }
        self._pages[page_id] = stored
        return stored

    @staticmethod
    def _page_brief(page: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "page_id": page["page_id"],
            "title": page["title"],
            "source": page["source"],
            "parent_title": page["parent_title"],
            "kind": page["kind"],
        }

    def search_company(self, corp_name_query: str, limit: int = 10) -> Dict[str, Any]:
        df = self.explorer.get_corp_codes()
        query = corp_name_query.strip()
        if not query:
            raise ValueError("corp_name_query is required")

        exact = df[df["corp_name"] == query]
        if not exact.empty:
            candidates = exact.head(limit)
        else:
            candidates = df[
                df["corp_name"].str.contains(query, case=False, na=False)
                | df["stock_code"].str.contains(query, case=False, na=False)
            ].head(limit)

        if candidates.empty:
            raise ValueError(f"회사 검색 결과가 없습니다: {corp_name_query}")

        items = []
        for row in candidates.to_dict(orient="records"):
            items.append(
                {
                    "corp_name": row.get("corp_name"),
                    "corp_code": row.get("corp_code"),
                    "stock_code": row.get("stock_code"),
                    "corp_eng_name": row.get("corp_eng_name"),
                }
            )
        return {"ok": True, "query": corp_name_query, "candidate_count": len(items), "candidates": items}

    def search_reports(
        self,
        corp_name: str,
        bgn_de: str,
        end_de: str,
        pblntf_ty: Optional[str] = None,
        pblntf_detail_ty: Optional[str] = None,
        last_reprt_at: Optional[str] = None,
        report_type: Optional[str] = None,
        report_name_query: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        corp = self.explorer.find_corp_exact(corp_name)
        normalized_pblntf_ty = str(
            pblntf_ty or self._infer_pblntf_ty(pblntf_detail_ty) or ""
        ).strip().upper() or None
        normalized_last_reprt_at = (
            str(last_reprt_at).strip().upper()
            if last_reprt_at is not None
            else self._default_last_reprt_at(pblntf_detail_ty, normalized_pblntf_ty)
        )
        normalized_report_name_query = self._normalize_report_name_query(
            pblntf_detail_ty,
            normalized_pblntf_ty,
            report_name_query,
        )
        if normalized_last_reprt_at not in {"Y", "N"}:
            raise ValueError("last_reprt_at must be 'Y' or 'N'")

        reports = self.explorer.search_filings(
            corp_code=corp["corp_code"],
            bgn_de=bgn_de,
            end_de=end_de,
            pblntf_ty=normalized_pblntf_ty,
            pblntf_detail_ty=pblntf_detail_ty,
            last_reprt_at=normalized_last_reprt_at,
            report_type=report_type,
            report_name_query=normalized_report_name_query,
        )
        if reports.empty:
            return {
                "ok": True,
                "corp_name": corp["corp_name"],
                "corp_code": corp["corp_code"],
                "search": {
                    "bgn_de": bgn_de,
                    "end_de": end_de,
                    "pblntf_ty": normalized_pblntf_ty,
                    "pblntf_ty_label": PBLNTF_TYPE_LABELS.get(normalized_pblntf_ty),
                    "pblntf_detail_ty": pblntf_detail_ty,
                    "pblntf_detail_ty_label": PBLNTF_DETAIL_TYPE_LABELS.get(pblntf_detail_ty),
                    "last_reprt_at": normalized_last_reprt_at,
                    "report_type": report_type,
                    "report_name_query": normalized_report_name_query,
                },
                "report_count": 0,
                "reports": [],
            }

        items: List[Dict[str, Any]] = []
        for row in reports.head(limit).to_dict(orient="records"):
            items.append(
                self._build_report_payload(
                    corp=corp,
                    report_row=row,
                    extra={
                        "search_window": {"bgn_de": bgn_de, "end_de": end_de},
                    },
                )
            )

        return {
            "ok": True,
            "corp_name": corp["corp_name"],
            "corp_code": corp["corp_code"],
            "search": {
                "bgn_de": bgn_de,
                "end_de": end_de,
                "pblntf_ty": normalized_pblntf_ty,
                "pblntf_ty_label": PBLNTF_TYPE_LABELS.get(normalized_pblntf_ty),
                "pblntf_detail_ty": pblntf_detail_ty,
                "pblntf_detail_ty_label": PBLNTF_DETAIL_TYPE_LABELS.get(pblntf_detail_ty),
                "last_reprt_at": normalized_last_reprt_at,
                "report_type": report_type,
                "report_name_query": normalized_report_name_query,
            },
            "report_count": len(items),
            "reports": items,
        }

    def search_recent_filings_by_stock_code(
        self,
        stock_code: str,
        bgn_de: Optional[str] = None,
        end_de: Optional[str] = None,
        pblntf_ty: Optional[str] = None,
        pblntf_detail_ty: Optional[str] = None,
        last_reprt_at: str = "Y",
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        종목코드 기준으로 최근 공시 목록을 넓게 조회한다.
        Broad recent-filings lookup by stock code. Use this when the user asks whether filings exist
        on a date, wants recent filing titles, or when the exact filing has not been determined yet.
        """
        corp = self.explorer.find_corp_by_stock_code(stock_code)
        today = dt.date.today()
        begin = str(bgn_de or f"{today.year - 1}0101").strip()
        end = str(end_de or today.strftime("%Y%m%d")).strip()
        reports = self.explorer.search_filings(
            corp_code=corp["corp_code"],
            bgn_de=begin,
            end_de=end,
            pblntf_ty=pblntf_ty,
            pblntf_detail_ty=pblntf_detail_ty,
            last_reprt_at=str(last_reprt_at or "Y").strip().upper() or "Y",
            report_type=None,
            report_name_query=None,
        )
        if reports.empty:
            return {
                "ok": True,
                "corp_name": corp["corp_name"],
                "corp_code": corp["corp_code"],
                "stock_code": corp.get("stock_code"),
                "report_count": 0,
                "reports": [],
                "search": {
                    "bgn_de": begin,
                    "end_de": end,
                    "pblntf_ty": pblntf_ty,
                    "pblntf_detail_ty": pblntf_detail_ty,
                    "last_reprt_at": str(last_reprt_at or "Y").strip().upper() or "Y",
                },
            }

        items: List[Dict[str, Any]] = []
        for row in reports.head(limit).to_dict(orient="records"):
            items.append(
                self._build_report_payload(
                    corp=corp,
                    report_row=row,
                    extra={"search_window": {"bgn_de": begin, "end_de": end}},
                )
            )
        return {
            "ok": True,
            "corp_name": corp["corp_name"],
            "corp_code": corp["corp_code"],
            "stock_code": corp.get("stock_code"),
            "report_count": len(items),
            "reports": items,
            "search": {
                "bgn_de": begin,
                "end_de": end,
                "pblntf_ty": pblntf_ty,
                "pblntf_detail_ty": pblntf_detail_ty,
                "last_reprt_at": str(last_reprt_at or "Y").strip().upper() or "Y",
            },
        }

    def find_latest_regular_report(
        self,
        corp_name: str,
        prefer_final_report: bool = True,
    ) -> Dict[str, Any]:
        corp = self.explorer.find_corp_exact(corp_name)
        today = dt.date.today()
        reports = self.explorer.search_filings(
            corp_code=corp["corp_code"],
            bgn_de=f"{today.year - 3}0101",
            end_de=today.strftime("%Y%m%d"),
            pblntf_detail_ty=None,
            last_reprt_at="Y" if prefer_final_report else "N",
            report_type="regular",
        )
        if reports.empty:
            raise ValueError(f"{corp_name}의 최근 정기보고서를 찾지 못했습니다.")

        report = reports.iloc[0].to_dict()
        payload = self._build_report_payload(
            corp=corp,
            report_row=report,
            extra={"prefer_final_report": prefer_final_report},
        )
        return {"ok": True, "report": payload}

    def find_business_report(
        self,
        corp_name: str,
        report_year: int,
        prefer_final_report: bool = True,
    ) -> Dict[str, Any]:
        corp = self.explorer.find_corp_exact(corp_name)
        bgn_de, end_de = self._report_window(report_year)

        reports = self.explorer.search_filings(
            corp_code=corp["corp_code"],
            bgn_de=bgn_de,
            end_de=end_de,
            pblntf_detail_ty="A001",
            last_reprt_at="Y" if prefer_final_report else "N",
        )
        if reports.empty:
            raise ValueError(f"{corp_name}의 {report_year} 사업보고서를 찾지 못했습니다.")

        report = reports.iloc[0].to_dict()
        payload = self._build_report_payload(
            corp=corp,
            report_row=report,
            extra={
                "report_year": report_year,
                "prefer_final_report": prefer_final_report,
                "search_window": {"bgn_de": bgn_de, "end_de": end_de},
            },
        )
        return {"ok": True, "report": payload}

    def get_report_archive_members(self, report_id: str) -> Dict[str, Any]:
        report = self._require_report(report_id)
        archive = self.explorer.fetch_document_archive(report["rcept_no"])
        return {"ok": True, "report": report, "archive": archive}

    def get_financial_statement_rows(
        self,
        report_id: str,
        statement_scope: str = "consolidated",
        max_rows: int = 300,
    ) -> Dict[str, Any]:
        report = self._require_report(report_id)
        if not report.get("business_year") or not report.get("reprt_code"):
            raise ValueError("selected report does not expose regular financial statement metadata")

        fs_div, scope_label = self.explorer._fs_div_from_scope(statement_scope)
        rows = self.explorer._fetch_financial_statement_rows(
            corp_code=report["corp_code"],
            bsns_year=report["business_year"],
            reprt_code=report["reprt_code"],
            fs_div=fs_div,
        )
        return {
            "ok": True,
            "report": report,
            "statement_scope": statement_scope,
            "scope_label": scope_label,
            "financial_rows": self.explorer._serialize_df(rows, max_rows=max_rows),
        }

    def get_financial_statement_key_contents(
        self,
        report_id: str,
        statement_scope: str = "consolidated",
        max_table_rows: int = 20,
        max_note_blocks: int = 30,
    ) -> Dict[str, Any]:
        report = self._require_report(report_id)
        contents = self.explorer.get_financial_statement_key_contents(
            corp_code=report["corp_code"],
            rcept_no=report["rcept_no"],
            statement_scope=statement_scope,
            max_table_rows=max_table_rows,
            max_note_blocks=max_note_blocks,
        )
        return {
            "ok": True,
            "report": report,
            "statement_scope": statement_scope,
            "markdown": contents,
        }

    def list_report_pages(
        self,
        report_id: str,
        title_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        report = self._require_report(report_id)
        pages = self.explorer.get_all_report_pages(report["rcept_no"])

        result_pages: List[Dict[str, Any]] = []
        for page in pages:
            if title_filter and title_filter not in page["title"]:
                continue

            stored = self._store_page(report_id=report_id, page=page)
            result_pages.append(self._page_brief(stored))

        return {
            "ok": True,
            "report": report,
            "page_count": len(result_pages),
            "pages": result_pages,
        }

    def search_report_pages(
        self,
        report_id: str,
        query: str,
        limit: int = 10,
        include_preview: bool = True,
        preview_max_blocks: int = 3,
    ) -> Dict[str, Any]:
        report = self._require_report(report_id)
        if not query or not query.strip():
            raise ValueError("query is required")

        pages = self.explorer.get_all_report_pages(report["rcept_no"])
        ranked: List[Dict[str, Any]] = []
        for page in pages:
            stored = self._store_page(report_id=report_id, page=page)
            title_score = self._score_query_match(query, stored["title"])
            parent_score = self._score_query_match(query, stored["parent_title"])
            kind_score = self._score_query_match(query, stored["kind"])
            total_score = title_score * 3 + parent_score * 2 + kind_score
            if total_score <= 0:
                continue
            ranked.append(
                {
                    "score": total_score,
                    "title_score": title_score,
                    "parent_score": parent_score,
                    "kind_score": kind_score,
                    "page": stored,
                }
            )

        ranked.sort(
            key=lambda item: (
                -item["score"],
                item["page"]["parent_title"] or "",
                item["page"]["title"],
            )
        )

        results: List[Dict[str, Any]] = []
        for item in ranked[:limit]:
            page = item["page"]
            payload = {
                **self._page_brief(page),
                "score": item["score"],
            }
            if include_preview:
                preview = self.explorer.extract_report_section(
                    url=page["url"],
                    page_title=page["title"],
                    whole_page=True,
                    max_table_rows=5,
                    max_blocks=max(1, preview_max_blocks),
                )
                preview_texts = [
                    block["content"]
                    for block in preview["blocks"]
                    if block["type"] == "text" and block.get("content")
                ]
                if preview_texts:
                    payload["preview"] = " ".join(preview_texts)[:400]
            results.append(payload)

        return {
            "ok": True,
            "report": report,
            "query": query,
            "page_count": len(results),
            "pages": results,
        }

    def list_page_subsections(self, page_id: str) -> Dict[str, Any]:
        page = self._require_page(page_id)
        sections = self.explorer.list_page_subsections(page["url"], page["title"])
        return {
            "ok": True,
            "page": self._page_brief(page),
            "section_count": len(sections),
            "sections": sections,
        }

    def search_page_subsections(
        self,
        page_id: str,
        query: str,
        limit: int = 10,
    ) -> Dict[str, Any]:
        page = self._require_page(page_id)
        if not query or not query.strip():
            raise ValueError("query is required")

        sections = self.explorer.list_page_subsections(page["url"], page["title"])
        ranked: List[Dict[str, Any]] = []
        for section in sections:
            score = self._score_query_match(query, section["title"])
            if score <= 0:
                continue
            ranked.append(
                {
                    "title": section["title"],
                    "toc_id": section["toc_id"],
                    "score": score,
                }
            )

        ranked.sort(key=lambda item: (-item["score"], item["title"]))
        return {
            "ok": True,
            "page": self._page_brief(page),
            "query": query,
            "section_count": len(ranked[:limit]),
            "sections": ranked[:limit],
        }

    def extract_report_section(
        self,
        page_id: str,
        keyword: Optional[str] = None,
        toc_id: Optional[str] = None,
        whole_page: bool = False,
        max_table_rows: int = 20,
        max_blocks: int = 80,
    ) -> Dict[str, Any]:
        page = self._require_page(page_id)
        result = self.explorer.extract_report_section(
            url=page["url"],
            page_title=page["title"],
            keyword=keyword,
            toc_id=toc_id,
            whole_page=whole_page,
            max_table_rows=max_table_rows,
            max_blocks=max_blocks,
        )
        return {
            "ok": True,
            "page": self._page_brief(page),
            **result,
        }

    def extract_report_sections(
        self,
        targets: List[Dict[str, Any]],
        max_table_rows: int = 20,
        max_blocks: int = 80,
    ) -> Dict[str, Any]:
        if not targets:
            raise ValueError("targets is required")

        results: List[Dict[str, Any]] = []
        for target in targets:
            page_id = target.get("page_id")
            if not page_id:
                results.append({"ok": False, "error": "page_id is required", "target": target})
                continue

            try:
                extracted = self.extract_report_section(
                    page_id=page_id,
                    keyword=target.get("keyword"),
                    toc_id=target.get("toc_id"),
                    whole_page=bool(target.get("whole_page", False)),
                    max_table_rows=max_table_rows,
                    max_blocks=max_blocks,
                )
                results.append(extracted)
            except Exception as e:
                results.append(
                    {
                        "ok": False,
                        "error": str(e),
                        "target": {
                            "page_id": page_id,
                            "keyword": target.get("keyword"),
                            "toc_id": target.get("toc_id"),
                            "whole_page": bool(target.get("whole_page", False)),
                        },
                    }
                )

        return {
            "ok": any(result.get("ok") for result in results),
            "result_count": len(results),
            "results": results,
        }

    def extract_note_section(
        self,
        report_id: str,
        note_scope: str,
        keyword: str,
        max_table_rows: int = 20,
        max_blocks: int = 80,
    ) -> Dict[str, Any]:
        report = self._require_report(report_id)

        pages_resp = self.list_report_pages(report_id=report_id)
        candidates = [
            p for p in pages_resp["pages"]
            if self._pick_note_scope(p["kind"], note_scope)
        ]
        if not candidates:
            raise ValueError(f"{note_scope} note page를 찾지 못했습니다.")

        target_page_id = candidates[0]["page_id"]
        extracted = self.extract_report_section(
            page_id=target_page_id,
            keyword=keyword,
            toc_id=None,
            whole_page=False,
            max_table_rows=max_table_rows,
            max_blocks=max_blocks,
        )
        return {
            "ok": True,
            "report": report,
            "note_scope": note_scope,
            "keyword": keyword,
            "page": extracted["page"],
            "mode": extracted["mode"],
            "matched_section_title": extracted["matched_section_title"],
            "matched_toc_id": extracted["matched_toc_id"],
            "available_section_count": extracted["available_section_count"],
            "blocks": extracted["blocks"],
        }


INTERNAL_TOOLS = [
    {
        "type": "function",
        "name": "search_company",
        "description": (
            "회사명 또는 종목코드로 상장사 후보를 찾는다. "
            "사용자가 회사코드를 직접 주지 않았거나 회사명이 애매하면 가장 먼저 호출한다. "
            "결과에는 corp_code, stock_code, 영문명이 포함된다."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "corp_name_query": {
                    "type": ["string", "null"],
                    "description": "회사명 또는 종목코드 일부. 예: 삼성전자, ISC, 005930",
                },
                "limit": {
                    "type": "integer",
                    "description": "최대 후보 개수. 보통 5~10이면 충분하다.",
                },
            },
            "required": ["corp_name_query", "limit"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "search_reports",
        "description": (
            "특정 회사의 공시 보고서 후보를 여러 건 찾는다. "
            "정기보고서만이 아니라 pblntf_detail_ty 분류코드 기준으로 주요사항보고서, 감사보고서, 수시공시 등도 찾을 수 있다. "
            "반환값에는 각 후보의 report_id, report_nm, 접수일, 사업연도 추정값, 정기보고서 reprt_code 추정값, 공시 분류코드가 포함된다. "
            "사용자는 이 결과를 보고 어느 연도/어느 보고서를 읽어야 하는지 선택할 수 있다."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "corp_name": {
                    "type": "string",
                    "description": "정확한 회사명. 애매하면 먼저 search_company로 후보를 찾는다.",
                },
                "bgn_de": {
                    "type": "string",
                    "description": "검색 시작일 YYYYMMDD. 예: 20240101",
                },
                "end_de": {
                    "type": "string",
                    "description": "검색 종료일 YYYYMMDD. 예: 20251231",
                },
                "pblntf_ty": {
                    "type": ["string", "null"],
                    "description": "DART 공시유형 대분류 코드. A~J. 비정기공시는 대분류와 상세코드를 같이 쓰는 편이 안전하다.",
                },
                "pblntf_detail_ty": {
                    "type": ["string", "null"],
                    "description": (
                        "DART 공시 상세유형 코드. 예: A001 사업보고서, A002 반기보고서, A003 분기보고서, "
                        "B001 주요사항보고서, E006 주주총회소집공고, F001 감사보고서, F002 연결감사보고서, "
                        "I001 수시공시. 전체를 넓게 찾을 때는 null."
                    ),
                },
                "last_reprt_at": {
                    "type": ["string", "null"],
                    "description": "최종보고서만 볼지 여부. 'Y' 또는 'N'. 정정 전 이력도 보면 'N'.",
                },
                "report_type": {
                    "type": ["string", "null"],
                    "description": (
                        "정기보고서 세부유형 보조필터. all/business/annual/semiannual/half/halfyear/"
                        "quarter/quarterly/regular 중 하나. 정기보고서가 아니면 보통 null."
                    ),
                },
                "report_name_query": {
                    "type": ["string", "null"],
                    "description": "report_nm 부분문자열 필터. 예: 합병, 소송, 자기주식, 감사보고서",
                },
                "limit": {
                    "type": "integer",
                    "description": "반환할 후보 수. 보통 5~20.",
                },
            },
            "required": [
                "corp_name",
                "bgn_de",
                "end_de",
                "pblntf_ty",
                "pblntf_detail_ty",
                "last_reprt_at",
                "report_type",
                "report_name_query",
                "limit",
            ],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "search_recent_filings_by_stock_code",
        "description": (
            "종목코드로 최근 공시 목록을 넓게 조회한다. "
            "특정 날짜에 공시가 나왔는지, 최근 공시 제목이 무엇인지, 최근 공시 현황을 먼저 확인해야 할 때 사용한다. "
            "이 요청은 보고서 본문 섹션 탐색이 아니라 list.json 기반 공시 목록 확인에 가깝다."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "stock_code": {"type": "string", "description": "6자리 종목코드. 예: 005930"},
                "bgn_de": {"type": ["string", "null"], "description": "검색 시작일 YYYYMMDD. 생략 시 최근 1년."},
                "end_de": {"type": ["string", "null"], "description": "검색 종료일 YYYYMMDD. 생략 시 오늘."},
                "pblntf_ty": {"type": ["string", "null"], "description": "대분류 공시 코드. 전체를 보려면 null."},
                "pblntf_detail_ty": {"type": ["string", "null"], "description": "상세 공시 코드. 전체를 보려면 null."},
                "last_reprt_at": {"type": "string", "description": "최종보고서만 보려면 Y, 이력 포함이면 N."},
                "limit": {"type": "integer", "description": "최대 반환 건수."},
            },
            "required": ["stock_code", "bgn_de", "end_de", "pblntf_ty", "pblntf_detail_ty", "last_reprt_at", "limit"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "find_latest_regular_report",
        "description": (
            "특정 회사의 가장 최근 정기보고서 1건을 찾는다. "
            "사업보고서, 반기보고서, 분기보고서 중 가장 최근 접수본을 반환한다. "
            "사용자가 '최근 보고서', '최신 공시', '최근 사업/분기보고서'처럼 연도를 명시하지 않았을 때 우선 호출한다."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "corp_name": {"type": "string", "description": "정확한 회사명. 예: 삼성전자"},
                "prefer_final_report": {"type": "boolean", "description": "최종 정정본 우선 여부"},
            },
            "required": ["corp_name", "prefer_final_report"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "find_business_report",
        "description": (
            "한국 회사의 특정 사업연도 사업보고서를 찾는다. "
            "사용자가 연도와 사업보고서를 명시했을 때 사용한다. "
            "예: 'ISC 2025년 사업보고서', '삼성전자 2024 사업보고서'."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "corp_name": {"type": "string", "description": "회사명. 예: ISC"},
                "report_year": {"type": "integer", "description": "사업연도. 예: 2025"},
                "prefer_final_report": {"type": "boolean", "description": "최종 정정본 우선 여부"},
            },
            "required": ["corp_name", "report_year", "prefer_final_report"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_report_archive_members",
        "description": (
            "공식 Open DART document.xml 원문 ZIP 안의 파일 목록을 본다. "
            "보고서 원문 구조를 확인하고 싶을 때 사용한다."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "report_id": {"type": "string"},
            },
            "required": ["report_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_financial_statement_rows",
        "description": (
            "선택한 정기보고서의 OpenDART 재무제표 행 데이터를 직접 가져온다. "
            "계산형 질문이나 계정과목 확인이 필요할 때 우선 사용한다. "
            "사업연도와 reprt_code가 있는 정기보고서에서만 동작한다."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "report_id": {"type": "string"},
                "statement_scope": {
                    "type": "string",
                    "enum": ["consolidated", "separate"],
                },
                "max_rows": {"type": "integer"},
            },
            "required": ["report_id", "statement_scope", "max_rows"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_financial_statement_key_contents",
        "description": (
            "선택한 정기보고서에 대해 재무제표 본문과 주요 주석 패키지를 한 번에 가져온다. "
            "재무제표 기반 요약, 계산, 주석 확인이 필요한 질문에서 shortcut으로 사용한다."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "report_id": {"type": "string"},
                "statement_scope": {
                    "type": "string",
                    "enum": ["consolidated", "separate"],
                },
                "max_table_rows": {"type": "integer"},
                "max_note_blocks": {"type": "integer"},
            },
            "required": ["report_id", "statement_scope", "max_table_rows", "max_note_blocks"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "list_report_pages",
        "description": (
            "사업보고서의 전체 페이지/하위문서 후보를 나열한다. "
            "먼저 어디를 볼지 탐색할 때 사용한다. "
            "직원, 임원, 보수, 사업의 내용, 재무제표, 주석 같은 제목 필터를 넣어 관련 페이지를 좁힐 수 있다."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "report_id": {"type": "string"},
                "title_filter": {
                    "type": ["string", "null"],
                    "description": "페이지 제목 필터. 예: 재무제표, 주석, 감사보고서, 사업의 내용",
                },
            },
            "required": ["report_id", "title_filter"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "list_page_subsections",
        "description": (
            "특정 페이지 안의 세부 목차(toc)를 나열한다. "
            "예: 연결재무제표 주석 페이지 안의 재고자산, 유형자산, 리스 등. "
            "사업의 내용이나 직원/임원 페이지 안에서 '직원 현황', '보수', '평균급여액' 같은 세부 구간을 찾을 때도 유용하다."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "page_id": {"type": "string"},
            },
            "required": ["page_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "search_report_pages",
        "description": (
            "선택한 보고서 안에서 질문과 관련 있을 가능성이 높은 페이지 후보를 질의문 기준으로 랭킹한다. "
            "페이지 제목, 상위 제목, kind를 점수화하고 필요하면 짧은 preview도 같이 준다. "
            "질문이 포괄적이어서 어느 페이지를 봐야 할지 애매할 때 먼저 사용한다."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "report_id": {"type": "string"},
                "query": {"type": "string"},
                "limit": {"type": "integer"},
                "include_preview": {"type": "boolean"},
                "preview_max_blocks": {"type": "integer"},
            },
            "required": ["report_id", "query", "limit", "include_preview", "preview_max_blocks"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "search_page_subsections",
        "description": (
            "특정 페이지 안의 세부 목차를 질의문 기준으로 랭킹한다. "
            "점수가 높은 toc_id 후보를 반환하므로, 이어서 extract_report_section이나 extract_report_sections에 넣어 읽으면 된다."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "page_id": {"type": "string"},
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["page_id", "query", "limit"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "extract_report_section",
        "description": (
            "특정 page_id에서 전체 페이지 또는 특정 목차 구간을 추출한다. "
            "keyword 또는 toc_id 중 하나로 특정 구간을 찾고, 둘 다 없으면 whole_page를 true로 줄 수 있다. "
            "직원 수, 평균 급여액, 임원 보수, 재무제표 본문처럼 보고서 본문에서 원하는 표/문단을 직접 찾을 때 쓴다."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "page_id": {"type": "string"},
                "keyword": {"type": ["string", "null"]},
                "toc_id": {"type": ["string", "null"]},
                "whole_page": {"type": "boolean"},
                "max_table_rows": {"type": "integer"},
                "max_blocks": {"type": "integer"},
            },
            "required": [
                "page_id",
                "keyword",
                "toc_id",
                "whole_page",
                "max_table_rows",
                "max_blocks",
            ],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "extract_report_sections",
        "description": (
            "여러 페이지/목차 위치를 한 번에 읽는다. "
            "한 질문에 대해 복수의 후보 위치를 동시에 확인하고 싶을 때 사용한다. "
            "각 target은 page_id와 함께 keyword 또는 toc_id 또는 whole_page를 줄 수 있다."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "targets": {
                    "type": "array",
                    "description": "읽고 싶은 위치 목록. 보통 2~5개 후보를 넣는다.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "page_id": {"type": "string"},
                            "keyword": {"type": ["string", "null"]},
                            "toc_id": {"type": ["string", "null"]},
                            "whole_page": {"type": "boolean"},
                        },
                        "required": ["page_id", "keyword", "toc_id", "whole_page"],
                        "additionalProperties": False,
                    },
                },
                "max_table_rows": {"type": "integer"},
                "max_blocks": {"type": "integer"},
            },
            "required": ["targets", "max_table_rows", "max_blocks"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "extract_note_section",
        "description": (
            "자주 쓰는 shortcut. 연결 또는 별도 재무제표 주석 페이지를 자동으로 찾은 뒤 "
            "특정 키워드 구간을 바로 추출한다."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "report_id": {"type": "string"},
                "note_scope": {
                    "type": "string",
                    "enum": ["consolidated", "separate"],
                },
                "keyword": {"type": "string"},
                "max_table_rows": {"type": "integer"},
                "max_blocks": {"type": "integer"},
            },
            "required": [
                "report_id",
                "note_scope",
                "keyword",
                "max_table_rows",
                "max_blocks",
            ],
            "additionalProperties": False,
        },
    },
]


TOOLS = [
    {
        "type": "function",
        "name": "find_dart_material",
        "description": (
            "Use this tool when the answer should be grounded in Korean DART disclosures rather than model memory or general finance knowledge. "
            "Call this tool especially when the user is asking about a specific Korean listed company, stock code, or legal entity and needs factual confirmation from filings. "
            "When deciding whether to call this tool, remember what each major filing family usually contains. "
            "A regular filings usually contain financial statements, business reports, operating results, management discussion, footnotes, dividends, financing and use of funds, governance, and related-company disclosures. "
            "B major-issues filings usually contain paid-in capital increases, bonds such as CB/BW/EB, treasury-stock actions, mergers, splits, major asset or business transfers, major contracts, and other major corporate actions. "
            "D ownership filings usually contain large shareholding changes, largest-shareholder changes, insider or executive holdings, and other ownership-change disclosures. "
            "E other disclosures often contain treasury-stock execution results, trust contracts, stock options, shareholder-meeting notices, outside-director changes, and other follow-up disclosure items. "
            "F external-audit filings contain audit opinions, audit reports, and audited financial statements and footnotes, and are especially useful for audit-opinion questions or for companies without regular business reports. "
            "This tool is a good fit in the following situations. "
            "1) Confirmed earnings or financial condition: for example revenue, operating profit, net income, debt ratio, cash flow, R&D expense, capex, or financial statement note language from quarterly reports, half-year reports, annual business reports, audit reports, or footnotes. "
            "2) Corporate actions: for example paid-in capital increase, bonus issue, CB, BW, bond issuance, dividend, treasury stock acquisition or disposal, stock split, reverse split, or cancellation. "
            "3) Ownership and governance changes: for example largest shareholder changes, 5 percent reports, insider trading or insider share purchases, executive shareholding changes, or post-dispute ownership structure changes. "
            "4) Risk, audit, contract, lawsuit, suspension, fraud, embezzlement, breach of trust, major supply contract, new investment, production halt, or listing-related events where the exact disclosure wording matters. "
            "5) Any request with recency or time sensitivity such as recent, latest, this quarter, this year, today, most recent filing, or based on the latest business report. "
            "Practical family hints: financial statements, footnotes, operating results, and MD&A usually point to A or F. Capital raises, bonds, mergers, splits, treasury stock, and major corporate actions usually point to B or E. Ownership-change questions usually point to D. Audit-opinion and auditor-wording questions usually point to F. "
            "In these cases, prefer DART lookup before answering from memory, especially when exact numbers, dates, terms, conditions, or source wording are required. "
            "Practical trigger rules: raise the priority of this tool if at least one of these is true: a specific Korean company name or stock code appears; the question requires the latest facts; the answer needs exact numbers, dates, or conditions; the user wants source paragraphs or evidence; or the request includes terms like earnings, revenue, operating profit, net income, debt ratio, rights offering, bonus issue, CB, BW, dividend, treasury shares, largest shareholder, insider, ownership change, contract, order, lawsuit, audit opinion, trading suspension, embezzlement, or breach of trust. "
            "Do not use this tool as the first choice for pure general knowledge or opinion questions such as what DART is, differences between report types, generic explanations of ROE or PER, how to read financial statements, whether a stock is good, whether the user should buy it, or why a share price moved. DART can still become supporting evidence later, but it is not the first tool for those questions. "
            "The input should be a natural-language retrieval request. Include the company name, and include the year, report type, disclosure topic, section name, accounting item, or keyword if the user already gave them. "
            "Dates and years are critical. Many users ask about the latest, recent, this year, last year, or a specific date. Think carefully about today's actual runtime date before calling this tool. "
            "If the user gives an explicit date, preserve that exact date in the tool query. Do not replace, reinterpret, or normalize the user's explicit date unless the user asked you to. "
            "For date-specific DART questions, pass the original date through unchanged. Do not decide whether a date is in the future or past from model memory. Judge it only against today's actual runtime date. "
            "The tool internally performs multi-step search and returns only the extracted source material after it is satisfied that the relevant filing passages were found. "
            "Return object: ok: whether relevant material was found. text: extracted source text from DART, mainly raw excerpt text with light whitespace cleanup. source_paths: array of source locations. Multiple entries may be returned if the material came from multiple sections or reports. Each item follows this format: [\"회사명_보고서연도_보고서시기_보고서내 항목명\", ...]. error: error message when ok is false."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural-language retrieval request for DART material. "
                        "Best practice: mention company name and what to find; if known, also mention year, report type, disclosure topic, section, note title, or accounting item. "
                        "Helpful filing-family hints: financial statements, business performance, footnotes, and MD&A usually belong to A or F; capital raises, bonds, mergers, splits, treasury stock, and other major corporate actions usually belong to B or E; ownership-change questions usually belong to D; audit-opinion questions usually belong to F. "
                        "If the user gave an explicit date, preserve that exact date in this query. Do not rewrite or normalize it on your own. "
                        "When the request is date-sensitive or uses words like recent/latest, think carefully about today's actual runtime date before constructing the query. "
                        "Examples: "
                        "'삼성전자 2024 사업보고서에서 유형자산 관련 감가상각비 설명 찾아줘', "
                        "'ISC 2025 사업보고서 연결재무제표 주석에서 유형자산 관련 내용 찾아줘', "
                        "'현대차 최근 사업보고서에서 리스 관련 위험 설명 문구 찾아줘'."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]


AGENT_INSTRUCTIONS = (
    "너는 대한민국 기업의 DART 공시에서 필요한 근거를 찾아 답하는 분석 도우미다. "
    "반드시 제공된 함수 결과에 근거해 답하고, 함수 결과에 없는 내용을 추정하지 않는다. "
    "회사가 애매하면 먼저 search_company로 회사 후보와 corp_code를 확인한다. "
    "어떤 보고서를 읽어야 할지 불분명하면 search_reports로 기간, 공시유형코드, 보고서명 키워드를 조절해 여러 후보를 찾는다. "
    "정기보고서가 명확할 때만 find_latest_regular_report 또는 find_business_report를 사용해도 된다. "
    "읽을 보고서를 정하면 먼저 search_report_pages 또는 list_report_pages로 관련 페이지 후보를 찾는다. "
    "페이지를 정한 뒤에는 search_page_subsections 또는 list_page_subsections로 세부 목차 후보를 좁힌다. "
    "그 다음 extract_report_section 또는 extract_report_sections로 본문과 표를 읽는다. "
    "한 번에 답을 단정하지 말고 필요한 경우 다른 페이지나 다른 목차 위치를 다시 골라 추가로 읽는다. "
    "처음 시도에서 부족하면 다른 위치를 고민해 2차 시도를 하고, 그래도 부족하면 다시 다른 위치로 3차 시도까지 한다. "
    "질문이 명백히 재무제표 주석이면 extract_note_section shortcut을 써도 된다. "
    "여러 위치를 확인했는데도 충분한 근거가 없으면 가능한 범위만 답하고, 어떤 보고서와 위치를 확인했지만 근거가 부족했는지 분명히 밝혀라."
)

AGENT_INSTRUCTIONS_V2 = (
    "너는 대한민국 기업의 DART 공시에서 필요한 근거를 찾아 답하는 분석 도우미다. "
    "반드시 제공된 함수 결과에 근거해 답하고, 함수 결과에 없는 내용을 추정하지 않는다. "
    "report_id, page_id, toc_id, 내부 JSON 같은 중간 식별자나 내부 산출물을 최종 답변에 노출하지 않는다. "
    "사용자에게 보고서 원문이나 데이터를 다시 보내달라고 먼저 요구하지 말고, 네가 가진 도구로 끝까지 확인한다. "
    "회사가 애매하면 먼저 search_company로 회사 후보와 corp_code를 확인한다. "
    "어떤 보고서를 읽어야 할지 불분명하면 search_reports로 기간, 공시유형코드, 보고서명 키워드를 조절해 여러 후보를 찾는다. "
    "정기보고서가 명확할 때만 find_latest_regular_report 또는 find_business_report를 사용해도 된다. "
    "질문이 재무제표 기반 계산이나 계정과목 확인이면, 보고서를 고른 직후 get_financial_statement_rows 또는 get_financial_statement_key_contents를 먼저 사용한다. "
    "읽을 보고서를 정하면 먼저 search_report_pages 또는 list_report_pages로 관련 페이지 후보를 찾는다. "
    "페이지를 정한 뒤에는 search_page_subsections 또는 list_page_subsections로 세부 목차 후보를 좁힌다. "
    "그 다음 extract_report_section 또는 extract_report_sections로 본문과 표를 읽는다. "
    "한 번에 답을 단정하지 말고 필요한 경우 다른 페이지나 다른 목차 위치를 다시 골라 추가로 읽는다. "
    "처음 시도에서 부족하면 다른 위치를 고민해 2차 시도를 하고, 그래도 부족하면 다시 다른 위치로 3차 시도까지 한다. "
    "질문이 계산형이면 가능한 범위에서 직접 계산해 답하고, 계산에 필요한 근거 항목을 스스로 더 찾는다. "
    "질문이 명백히 재무제표 주석이면 extract_note_section shortcut을 써도 된다. "
    "여러 위치를 확인했는데도 충분한 근거가 없으면 가능한 범위만 답하고, 어떤 보고서와 위치를 확인했지만 근거가 부족했는지 분명히 밝혀라."
)


def _sanitize_schema_for_gemini(schema: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}

    for key, value in schema.items():
        if key == "additionalProperties":
            continue

        if key == "type" and isinstance(value, list):
            non_null_types = [item for item in value if item != "null"]
            if len(non_null_types) == 1:
                sanitized["type"] = non_null_types[0]
                sanitized["nullable"] = "null" in value
                continue

        if key in {"properties", "items"}:
            if isinstance(value, dict):
                sanitized[key] = {
                    sub_key: _sanitize_schema_for_gemini(sub_value)
                    if isinstance(sub_value, dict)
                    else sub_value
                    for sub_key, sub_value in value.items()
                }
            else:
                sanitized[key] = value
            continue

        if isinstance(value, dict):
            sanitized[key] = _sanitize_schema_for_gemini(value)
        elif isinstance(value, list):
            sanitized[key] = [
                _sanitize_schema_for_gemini(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = value

    return sanitized


def _convert_openai_tools_to_gemini(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    declarations = []
    for tool in tools:
        declarations.append(
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": _sanitize_schema_for_gemini(tool["parameters"]),
            }
        )
    return [{"functionDeclarations": declarations}]


INTERNAL_GEMINI_TOOLS = _convert_openai_tools_to_gemini(INTERNAL_TOOLS)
PUBLIC_GEMINI_TOOLS = _convert_openai_tools_to_gemini(TOOLS)
TOOLS_BY_NAME = {tool["name"]: tool for tool in INTERNAL_TOOLS}


class BaseDartAgent:
    PUBLIC_TOOL_NAME = "find_dart_material"
    CORP_CLS_PRIORITY = {"Y": 0, "K": 1, "N": 2, "E": 3}
    EVIDENCE_RENDER_LIMIT = 24000
    LONG_EVIDENCE_THRESHOLD = 12000
    EVIDENCE_CHUNK_SIZE = 5000
    EVIDENCE_CHUNK_OVERLAP = 500
    FOCUSED_EVIDENCE_LIMIT = 24000
    COMPLEXITY_HINTS = (
        "왜", "원인", "비교", "추세", "리스크", "위험", "영향", "평가", "분석", "종합", "요약",
        "explain", "why", "compare", "trend", "risk", "analy", "summar", "impact"
    )
    FINAL_REWRITE_PROMPT = (
        "최종 답변만 다시 작성하라. "
        "report_id, page_id, toc_id, JSON 같은 내부 식별자와 중간 산출물은 노출하지 마라. "
        "사용자에게 추가 자료를 보내달라고 요구하지 마라. "
        "이미 확인한 도구 결과 범위에서만 답하고, 근거가 부족하면 무엇을 확인했는지 짧게 밝힌 뒤 가능한 범위의 결론만 제시하라."
    )
    CONTINUE_PROMPT = (
        "여기서 멈추지 말고 계속 진행하라. "
        "사용자에게 추가 자료를 요구하지 말고, 남은 도구를 사용해 필요한 보고서 페이지와 재무제표/주석을 더 찾아 계산 또는 답변을 완료하라. "
        "내부 식별자는 답변에 노출하지 마라."
    )
    NUMERIC_QUERY_HINTS = (
        "계산", "금액", "얼마", "감가상각", "상각", "비용", "합계", "연간", "연도", "비현금성",
        "calculate", "amount", "cost", "sum", "total", "depreciation", "amortization"
    )
    COLLECTION_ROUND_LIMIT = 12

    def __init__(
        self,
        dart_api_key: str,
        model: str,
    ):
        self.tool_service = DartReportToolService(dart_api_key=dart_api_key)
        self.tool_registry = {
            "search_company": self.tool_service.search_company,
            "search_reports": self.tool_service.search_reports,
            "search_recent_filings_by_stock_code": self.tool_service.search_recent_filings_by_stock_code,
            "find_latest_regular_report": self.tool_service.find_latest_regular_report,
            "find_business_report": self.tool_service.find_business_report,
            "get_report_archive_members": self.tool_service.get_report_archive_members,
            "get_financial_statement_rows": self.tool_service.get_financial_statement_rows,
            "get_financial_statement_key_contents": self.tool_service.get_financial_statement_key_contents,
            "list_report_pages": self.tool_service.list_report_pages,
            "search_report_pages": self.tool_service.search_report_pages,
            "list_page_subsections": self.tool_service.list_page_subsections,
            "search_page_subsections": self.tool_service.search_page_subsections,
            "extract_report_section": self.tool_service.extract_report_section,
            "extract_report_sections": self.tool_service.extract_report_sections,
            "extract_note_section": self.tool_service.extract_note_section,
        }
        self.instructions = AGENT_INSTRUCTIONS_V2
        self.model = model
        self._report_resolution_cache: Dict[Tuple[str, str, str, str, str, str, str], Dict[str, Any]] = {}
        self._toc_catalog_cache: Dict[str, Dict[str, Any]] = {}
        self._evidence_cache: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

    def _infer_pblntf_ty(self, pblntf_detail_ty: Optional[str]) -> Optional[str]:
        return self.tool_service._infer_pblntf_ty(pblntf_detail_ty)

    def _default_last_reprt_at(self, pblntf_detail_ty: Optional[str], pblntf_ty: Optional[str]) -> str:
        return self.tool_service._default_last_reprt_at(pblntf_detail_ty, pblntf_ty)

    def _normalize_report_name_query(
        self,
        pblntf_detail_ty: Optional[str],
        pblntf_ty: Optional[str],
        report_name_query: Optional[str],
    ) -> Optional[str]:
        return self.tool_service._normalize_report_name_query(
            pblntf_detail_ty,
            pblntf_ty,
            report_name_query,
        )

    def _log(self, title: str, detail: Any, kind: str = "info") -> None:
        logger = getattr(self, "push_log", None)
        if callable(logger):
            logger(title, detail, kind=kind)

    @staticmethod
    def _normalize_plan_key_text(value: Any) -> str:
        return str(value or "").strip().upper()

    def _dedupe_plans(self, plans: List[Dict[str, Any]], max_plans: int = 6) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for plan in plans:
            candidate_key = "|".join(
                self._normalize_plan_key_text(item)
                for item in (plan.get("corp_name_candidates") or [])
                if self._normalize_plan_key_text(item)
            )
            key = (
                self._normalize_plan_key_text(plan.get("corp_name_query")),
                candidate_key,
                self._normalize_plan_key_text(plan.get("business_year")),
                self._normalize_plan_key_text(plan.get("pblntf_ty")),
                self._normalize_plan_key_text(plan.get("pblntf_detail_ty")),
                self._normalize_plan_key_text(plan.get("reprt_code")),
                self._normalize_plan_key_text(plan.get("fs_div")),
                self._normalize_plan_key_text(plan.get("report_name_query")),
                self._normalize_plan_key_text(plan.get("request_mode")),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(plan)
            if len(deduped) >= max_plans:
                break
        return deduped

    @staticmethod
    def _plan_scope_query(user_text: str, plan: Dict[str, Any]) -> str:
        parts = [
            str(plan.get("plan_goal") or "").strip(),
            str(plan.get("corp_name_query") or "").strip(),
            " ".join(str(item).strip() for item in (plan.get("corp_name_candidates") or []) if str(item).strip()),
            str(plan.get("business_year") or "").strip(),
            str(plan.get("report_name_query") or "").strip(),
            str(plan.get("request_mode") or "").strip(),
            str(user_text or "").strip(),
        ]
        merged = " ".join(part for part in parts if part)
        if BaseDartAgent._has_recent_reference(merged):
            return BaseDartAgent._request_text_with_runtime_date(merged)
        return merged

    @staticmethod
    def _plan_scope_summary(plan: Dict[str, Any], corp: Optional[Dict[str, Any]] = None, report: Optional[Dict[str, Any]] = None) -> str:
        lines = [
            f"- plan_goal: {plan.get('plan_goal')}",
            f"- corp_name_query: {plan.get('corp_name_query')}",
            f"- corp_name_candidates: {plan.get('corp_name_candidates')}",
            f"- business_year: {plan.get('business_year')}",
            f"- pblntf_ty: {plan.get('pblntf_ty')}",
            f"- pblntf_detail_ty: {plan.get('pblntf_detail_ty')}",
            f"- reprt_code: {plan.get('reprt_code')}",
            f"- fs_div: {plan.get('fs_div')}",
            f"- request_mode: {plan.get('request_mode')}",
        ]
        if corp:
            lines.append(f"- resolved_company: {corp.get('corp_name')} ({corp.get('stock_code') or 'no stock code'})")
        if report:
            lines.append(f"- report_nm: {report.get('report_nm')}")
            lines.append(f"- rcept_dt: {report.get('rcept_dt')}")
        return "\n".join(lines)

    def _call_model_text(
        self,
        prompt: str,
        *,
        system_instruction: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        raise NotImplementedError

    @staticmethod
    def _extract_json_payload(text: str) -> Dict[str, Any]:
        raw = (text or "").strip()
        candidates = [raw]

        fenced = re.findall(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
        candidates.extend(item.strip() for item in fenced if item.strip())

        for opener, closer in (("{", "}"), ("[", "]")):
            start = raw.find(opener)
            end = raw.rfind(closer)
            if start != -1 and end != -1 and end > start:
                candidates.append(raw[start:end + 1])

        for candidate in candidates:
            if not candidate:
                continue
            try:
                payload = json.loads(candidate)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload

        raise ValueError(f"LLM did not return valid JSON object: {raw[:1000]}")

    def _ask_json(self, prompt: str, *, phase: str) -> Dict[str, Any]:
        self._log(
            "LLM request",
            {"phase": phase, "model": self.model},
            kind="model",
        )
        raw = self._call_model_text(prompt, json_mode=True)
        self._log("LLM response", {"phase": phase, "text": raw[:4000]}, kind="llm-response")
        return self._extract_json_payload(raw)

    @staticmethod
    def _report_code_reference_text() -> str:
        regular_lines = [
            "Regular reprt_code values:",
            "- 11013: 1분기보고서",
            "- 11012: 반기보고서",
            "- 11014: 3분기보고서",
            "- 11011: 사업보고서",
            "",
            "pblntf_ty values:",
        ]
        family_lines = [
            f"- {code}: {label}"
            for code, label in sorted(PBLNTF_TYPE_LABELS.items())
        ]
        detail_header = [
            "",
            "pblntf_detail_ty values:",
        ]
        detail_lines = [
            f"- {code}: {label}"
            for code, label in sorted(PBLNTF_DETAIL_TYPE_LABELS.items())
        ]
        return "\n".join(regular_lines + family_lines + detail_header + detail_lines)

    @staticmethod
    def _today_date() -> dt.date:
        return dt.date.today()

    @classmethod
    def _planning_time_context_text(cls) -> str:
        today = cls._today_date()
        latest_completed_year_end = today.year - 1
        return (
            f"Today's actual date is {today.isoformat()}.\n"
            "Interpret all relative time expressions using this actual runtime date, not the model's knowledge cutoff.\n"
            f"The latest completed fiscal year-end is {latest_completed_year_end}-12-31.\n"
            f"For example, if the user asks about 'recent year-end', 'latest year-end', or 'most recent year-end', "
            f"prefer business_year={latest_completed_year_end} unless the user explicitly asks for another year.\n"
            "If the user says 'last year', use the year immediately before today's year. "
            "If the user says 'this year', use today's year."
        )

    @staticmethod
    def _has_recent_reference(text: Any) -> bool:
        lowered = str(text or "").lower()
        return any(marker.lower() in lowered for marker in RECENCY_MARKERS)

    @classmethod
    def _request_text_with_runtime_date(cls, text: Any) -> str:
        value = str(text or "").strip()
        if not value:
            return value
        runtime_line = f"Today's actual date: {cls._today_date().isoformat()}"
        if runtime_line in value:
            return value
        if cls._has_recent_reference(value):
            return f"{value}\n\n{runtime_line}"
        return value

    @classmethod
    def _normalize_relative_business_year(cls, user_text: str, proposed_year: Any) -> int:
        fallback = cls._infer_business_year_fallback(user_text)
        try:
            year = int(str(proposed_year or fallback)[:4])
        except Exception:
            return fallback
        if re.search(r"\b20\d{2}\b", str(user_text or "")):
            return year
        if cls._has_recent_reference(user_text) and year < (fallback - 1):
            return fallback
        return year

    @staticmethod
    def _normalize_request_mode(user_text: str, raw_mode: Any) -> str:
        mode = str(raw_mode or "").strip().lower()
        if mode in {"section_text", "filing_list", "filing_existence"}:
            return mode
        text = str(user_text or "").lower()
        existence_markers = ("공시 나왔", "공시 있어", "공시 있었", "나왔어", "있어?", "있나", "filed on", "was there a filing")
        list_markers = ("공시 목록", "공시 현황", "최근 공시", "최신 공시", "무슨 공시", "어떤 공시", "공시 제목", "filing list", "recent filings")
        if any(marker in text for marker in existence_markers):
            return "filing_existence"
        if any(marker in text for marker in list_markers):
            return "filing_list"
        return "section_text"

    @staticmethod
    def _topic_resolution_candidates(user_text: str, plan: Dict[str, Any]) -> List[Tuple[Optional[str], Optional[str], Optional[str]]]:
        text = " ".join(
            str(value or "")
            for value in [user_text, plan.get("plan_goal"), plan.get("report_name_query"), plan.get("reason")]
        ).lower()
        candidates: List[Tuple[Optional[str], Optional[str], Optional[str]]] = []
        if "배당" in text or "dividend" in text:
            candidates.extend([("B", "B001", "배당"), ("A", "A001", "배당")])
        if any(marker in text for marker in ("자사주", "자기주식", "treasury")):
            candidates.append(("E", "E001", "자기주식"))
        if any(marker in text for marker in ("지분변동", "최대주주", "5%", "임원 지분", "ownership change")):
            candidates.extend([("D", "D001", None), ("D", "D002", None)])
        if any(marker in text for marker in ("감사의견", "감사보고서", "audit opinion", "audit report")):
            candidates.extend([("F", "F002", None), ("F", "F001", None), ("F", "F003", None)])
        deduped: List[Tuple[Optional[str], Optional[str], Optional[str]]] = []
        seen = set()
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    @staticmethod
    def _filing_family_guidance_text() -> str:
        return (
            "Filing-family routing guidance:\n"
            "- A regular filings include annual business reports, half-year reports, quarterly reports, and older regular settlement documents. They usually contain company overview, business description, MD&A, summary financials, consolidated and separate financial statements, footnotes, dividend information, financing and use of funds, governance, largest shareholder, board, shareholder meeting, affiliate information, and internal-control related attachments.\n"
            "- Use A first when the user asks about financial statements, operating results, revenue, operating profit, net income, debt ratio, cash flow, capex, R&D, business description, MD&A, footnotes, dividend details inside regular reports, or management explanations of performance changes.\n"
            "- B major issues reports usually contain bankruptcy/default, business suspension, rehabilitation proceedings, dissolution events, capital increases or decreases, CB/BW/EB and similar securities issuance decisions, treasury stock acquisition or disposal, treasury trust contracts, mergers, stock swaps, transfers, spin-offs, important business or asset transfers, major lawsuits, overseas listing or delisting events, and major contracts such as put-back or similar arrangements.\n"
            "- Use B first when the user asks about rights offerings, paid-in capital increases, convertible bonds, bonds with warrants, exchangeable bonds, major mergers or splits, major asset sales or acquisitions, major contracts, suspension events, insolvency-related events, or other major corporate actions.\n"
            "- C issuance filings usually contain securities registration details, offering structure, rights of the securities, risk factors, use of proceeds, issuer information, and issuance schedules or results. Consider C when the user is asking about prospectus-style offering details rather than ordinary post-event disclosures.\n"
            "- D ownership filings usually contain 5 percent reports, large-shareholding changes, tender offers, proxy solicitations, insider or executive holdings, major shareholder holdings, ownership purpose, related parties, pledge or trust or lending arrangements, and planned transactions by insiders or major shareholders.\n"
            "- Use D first when the user asks about largest-shareholder changes, 5 percent holdings, insider trading or insider share purchases, executive shareholding changes, tender offers, or changes in ownership structure.\n"
            "- E other disclosures usually contain treasury stock execution results, trust-contract execution or termination, merger completion reports, stock option grants, outside-director changes, shareholder-meeting notices, market making or stabilization actions, covered bonds, and similar follow-up disclosure items.\n"
            "- Use E especially for treasury-stock execution results, trust-contract results, shareholder-meeting notices, outside-director filings, stock-option grant disclosures, and other follow-up or completion disclosures that are not best represented by A or B.\n"
            "- F external-audit related filings include audit reports, consolidated audit reports, combined audit reports, auditor opinion wording, audited financial statements and footnotes, accounting-firm reports, and pre-audit financial-statement non-submission notices.\n"
            "- Use F first when the user asks about audit opinions, qualified/adverse/disclaimer issues, auditor findings, wording from audit reports, or when the target company may not have ordinary business reports but still has audited financial statements and footnotes.\n"
            "- Fund disclosures and securitization disclosures exist too, but they are usually not the first choice unless the user is explicitly asking about fund prospectus content, asset-backed securities, securitization plans, transfer registrations, or related issuance structures.\n"
            "- Dividend questions may require A and sometimes B or E together.\n"
            "- If the user asks for filing existence, filing titles, filing history, or what was filed on a date, prefer filing_list or filing_existence rather than forcing a body-text plan."
        )

    @staticmethod
    def _extract_explicit_date_window(user_text: str) -> Optional[Tuple[str, str]]:
        text = str(user_text or "").strip()
        if not text:
            return None
        month_range_match = re.search(
            r"(20\d{2})\s*년\s*(\d{1,2})\s*월\s*(?:~|부터|-)\s*(20\d{2})\s*년\s*(\d{1,2})\s*월",
            text,
        )
        if month_range_match:
            start_year, start_month, end_year, end_month = map(int, month_range_match.groups())
            start_date = dt.date(start_year, start_month, 1)
            end_day = calendar.monthrange(end_year, end_month)[1]
            end_date = dt.date(end_year, end_month, end_day)
            return start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")
        dotted_month_range_match = re.search(
            r"\b(20\d{2})[./-](\d{1,2})\s*(?:~|to|-)\s*(20\d{2})[./-](\d{1,2})\b",
            text,
        )
        if dotted_month_range_match:
            start_year, start_month, end_year, end_month = map(int, dotted_month_range_match.groups())
            start_date = dt.date(start_year, start_month, 1)
            end_day = calendar.monthrange(end_year, end_month)[1]
            end_date = dt.date(end_year, end_month, end_day)
            return start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")
        match = re.search(r"(20\d{2})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", text)
        if match:
            y, m, d = map(int, match.groups())
            date_value = dt.date(y, m, d)
            day = date_value.strftime("%Y%m%d")
            return day, day
        match = re.search(r"\b(20\d{2})[-./](\d{1,2})[-./](\d{1,2})\b", text)
        if match:
            y, m, d = map(int, match.groups())
            date_value = dt.date(y, m, d)
            day = date_value.strftime("%Y%m%d")
            return day, day
        match = re.search(r"\b(20\d{2})(\d{2})(\d{2})\b", text)
        if match:
            y, m, d = map(int, match.groups())
            date_value = dt.date(y, m, d)
            day = date_value.strftime("%Y%m%d")
            return day, day
        return None

    @staticmethod
    def _shift_date_by_months(date_value: dt.date, months: int) -> dt.date:
        total_months = date_value.year * 12 + (date_value.month - 1) - months
        year = total_months // 12
        month = total_months % 12 + 1
        day = min(date_value.day, calendar.monthrange(year, month)[1])
        return dt.date(year, month, day)

    @classmethod
    def _extract_relative_period_window(cls, user_text: str) -> Optional[Tuple[str, str]]:
        text = str(user_text or "").strip().lower()
        if not text:
            return None
        today = cls._today_date()
        month_match = re.search(r"(?:최근|지난)\s*(\d+)\s*(?:개?\s*월|달)(?:간)?", text)
        if month_match:
            months = int(month_match.group(1))
            if months > 0:
                start_date = cls._shift_date_by_months(today, months)
                return start_date.strftime("%Y%m%d"), today.strftime("%Y%m%d")
        year_match = re.search(r"(?:최근|지난)\s*(\d+)\s*(?:개?\s*년)(?:간)?", text)
        if year_match:
            years = int(year_match.group(1))
            if years > 0:
                start_date = cls._shift_date_by_months(today, years * 12)
                return start_date.strftime("%Y%m%d"), today.strftime("%Y%m%d")
        english_month_match = re.search(r"(?:recent|last)\s*(\d+)\s*months?", text)
        if english_month_match:
            months = int(english_month_match.group(1))
            if months > 0:
                start_date = cls._shift_date_by_months(today, months)
                return start_date.strftime("%Y%m%d"), today.strftime("%Y%m%d")
        english_year_match = re.search(r"(?:recent|last)\s*(\d+)\s*years?", text)
        if english_year_match:
            years = int(english_year_match.group(1))
            if years > 0:
                start_date = cls._shift_date_by_months(today, years * 12)
                return start_date.strftime("%Y%m%d"), today.strftime("%Y%m%d")
        return None

    @classmethod
    def _infer_business_year_fallback(cls, user_text: str) -> int:
        today = cls._today_date()
        text = str(user_text or "").lower()

        explicit_year = re.search(r"\b(20\d{2})\b", text)
        if explicit_year:
            return int(explicit_year.group(1))

        recent_year_end_markers = (
            "최근 연말", "최신 연말", "가장 최근 연말", "최근 결산", "최신 결산", "연말 기준",
            "recent year-end", "latest year-end", "most recent year-end", "year-end",
        )
        if any(marker in text for marker in recent_year_end_markers):
            return today.year - 1

        last_year_markers = ("작년", "전년", "지난해", "last year", "previous year")
        if any(marker in text for marker in last_year_markers):
            return today.year - 1

        this_year_markers = ("올해", "금년", "this year", "current year")
        if any(marker in text for marker in this_year_markers):
            return today.year

        return today.year

    @staticmethod
    def _compact_text(value: Any, limit: int = 4000) -> str:
        text = str(value or "").strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        if len(text) <= limit:
            return text
        return text[:limit] + "\n... (truncated)"

    def _candidate_company_queries(self, user_text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9가-힣][A-Za-z0-9가-힣&()./-]{0,30}", user_text or "")
        stopwords = {
            "dart", "DART", "api", "API", "함수", "통해", "찾아줘", "계산해줘", "최근", "가장",
            "기준", "연간", "보고서", "사업보고서", "반기보고서", "분기보고서", "연도", "년도",
            "감가상각액", "비현금성비용", "연결", "개별",
        }
        queries: List[str] = []
        for token in tokens[:12]:
            if token in stopwords:
                continue
            if token.lower() in {word.lower() for word in stopwords}:
                continue
            queries.append(token)
            upper = token.upper()
            if upper != token:
                queries.append(upper)

        deduped: List[str] = []
        for query in queries:
            if query and query not in deduped:
                deduped.append(query)
        return deduped[:10]

    @classmethod
    def _corp_cls_rank(cls, corp_cls: Any) -> int:
        return cls.CORP_CLS_PRIORITY.get(str(corp_cls or "").strip().upper(), 9)

    def _load_corp_cls(self, corp_code: Any) -> str:
        corp_code_text = str(corp_code or "").strip()
        if not corp_code_text:
            return ""
        try:
            overview = self.tool_service.explorer.get_company_overview(corp_code_text)
        except Exception:
            return ""
        return str(overview.get("corp_cls") or "").strip().upper()

    def _rank_exact_company_candidates(
        self,
        plan: Dict[str, Any],
        matches: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        candidate_order = {
            str(name).strip(): index
            for index, name in enumerate(plan.get("corp_name_candidates") or [])
            if str(name).strip()
        }

        ranked: List[Tuple[Tuple[int, int, int, str], Dict[str, Any]]] = []
        for match in matches:
            corp_cls = self._load_corp_cls(match.get("corp_code"))
            candidate = dict(match)
            candidate["corp_cls"] = corp_cls
            matched_name = str(match.get("matched_candidate") or "").strip()
            name_order = candidate_order.get(matched_name, 999)
            ranked.append(
                (
                    (
                        self._corp_cls_rank(corp_cls),
                        name_order,
                        0 if str(candidate.get("stock_code") or "").strip() else 1,
                        str(candidate.get("corp_name") or ""),
                    ),
                    candidate,
                )
            )

        ranked.sort(key=lambda item: item[0])
        return [candidate for _, candidate in ranked]

    def _collect_exact_company_candidates(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidate_names: List[str] = []
        for value in (plan.get("corp_name_candidates") or []):
            text = str(value or "").strip()
            if text and text not in candidate_names:
                candidate_names.append(text)

        corp_name_query = str(plan.get("corp_name_query") or "").strip()
        if corp_name_query and corp_name_query not in candidate_names:
            candidate_names.append(corp_name_query)

        if not candidate_names:
            return []

        df = self.tool_service.explorer.get_corp_codes()
        matches: List[Dict[str, Any]] = []
        seen = set()
        for candidate_name in candidate_names:
            exact_rows = df[df["corp_name"] == candidate_name]
            if exact_rows.empty:
                continue
            for row in exact_rows.to_dict(orient="records"):
                key = (
                    str(row.get("corp_code") or "").strip(),
                    str(candidate_name).strip(),
                )
                if key in seen:
                    continue
                seen.add(key)
                matches.append(
                    {
                        "corp_name": row.get("corp_name"),
                        "corp_code": row.get("corp_code"),
                        "stock_code": row.get("stock_code"),
                        "corp_eng_name": row.get("corp_eng_name"),
                        "matched_candidate": candidate_name,
                    }
                )
        return self._rank_exact_company_candidates(plan, matches)

    def _pick_company_candidate(self, query: str, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        best_candidate = None
        best_score = 0
        for candidate in candidates:
            score = 0
            score += self.tool_service._score_query_match(query, candidate.get("corp_name")) * 5
            score += self.tool_service._score_query_match(query, candidate.get("corp_eng_name")) * 2
            score += self.tool_service._score_query_match(query, candidate.get("stock_code")) * 6
            if str(candidate.get("corp_name", "")).lower() == query.lower():
                score += 5000
            if str(candidate.get("stock_code", "")).lower() == query.lower():
                score += 5000
            corp_cls = str(candidate.get("corp_cls") or "").strip().upper()
            if corp_cls:
                score += max(0, 40 - (self._corp_cls_rank(corp_cls) * 10))
            if score > best_score:
                best_candidate = candidate
                best_score = score
        return best_candidate if best_score > 0 else None

    @staticmethod
    def _normalize_company_query_text(value: Any) -> str:
        text = OpenDartReportExplorer._normalize_text(str(value or "")).strip()
        text = re.sub(r"^\(?\s*(?:주식회사|㈜|\(주\)|주\))\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*\(?\s*(?:주식회사|㈜|\(주\)|주\))\s*$", "", text, flags=re.IGNORECASE)
        return text.strip()

    @classmethod
    def _normalize_company_query_key(cls, value: Any) -> str:
        text = cls._normalize_company_query_text(value).lower()
        return re.sub(r"[^0-9a-z가-힣]+", "", text)

    @staticmethod
    def _english_letters_to_korean(text: str) -> str:
        letter_map = {
            "A": "에이", "B": "비", "C": "씨", "D": "디", "E": "이", "F": "에프", "G": "지",
            "H": "에이치", "I": "아이", "J": "제이", "K": "케이", "L": "엘", "M": "엠", "N": "엔",
            "O": "오", "P": "피", "Q": "큐", "R": "알", "S": "에스", "T": "티", "U": "유",
            "V": "브이", "W": "더블유", "X": "엑스", "Y": "와이", "Z": "지",
        }
        return "".join(letter_map.get(ch, ch) for ch in str(text or "").upper())

    @classmethod
    def _expand_company_query_variants(cls, query: str) -> List[str]:
        query_text = cls._normalize_company_query_text(query)
        if not query_text:
            return []

        variants: List[str] = []

        def add(value: Any) -> None:
            text = cls._normalize_company_query_text(value)
            if text and text not in variants:
                variants.append(text)

        add(query_text)
        add(query_text.replace(" ", ""))

        acronym_match = re.match(r"^([A-Z]{2,})(.*)$", query_text)
        if acronym_match:
            acronym = acronym_match.group(1)
            tail = acronym_match.group(2)
            expanded = cls._english_letters_to_korean(acronym) + tail
            add(expanded)
            add(expanded.replace(" ", ""))

        for token_match in re.finditer(r"[A-Z]{2,}", query_text):
            token = token_match.group(0)
            expanded_token = cls._english_letters_to_korean(token)
            add(query_text.replace(token, expanded_token))
            add(query_text.replace(token, expanded_token).replace(" ", ""))

        return variants

    def _candidate_company_attempts(self, plan: Dict[str, Any], max_attempts: int = 10) -> List[str]:
        seeds: List[str] = []
        for candidate in (plan.get("corp_name_candidates") or []):
            candidate_text = str(candidate or "").strip()
            if candidate_text and candidate_text not in seeds:
                seeds.append(candidate_text)

        corp_name_query = str(plan.get("corp_name_query") or "").strip()
        if corp_name_query and corp_name_query not in seeds:
            seeds.append(corp_name_query)

        attempts: List[str] = []
        for seed in seeds:
            for variant in self._expand_company_query_variants(seed):
                if variant and variant not in attempts:
                    attempts.append(variant)
                if len(attempts) >= max_attempts:
                    return attempts
        return attempts[:max_attempts]

    def _search_company_by_normalized_key(self, corp_name_query: str, limit: int = 5) -> List[Dict[str, Any]]:
        query_key = self._normalize_company_query_key(corp_name_query)
        if not query_key:
            return []

        df = self.tool_service.explorer.get_corp_codes()
        ranked: List[Tuple[int, Dict[str, Any]]] = []
        for row in df.to_dict(orient="records"):
            corp_name = row.get("corp_name")
            corp_name_key = self._normalize_company_query_key(corp_name)
            stock_code = str(row.get("stock_code") or "").strip()

            score = 0
            if corp_name_key == query_key:
                score += 10000
            elif query_key and corp_name_key and query_key in corp_name_key:
                score += 7000
            elif query_key and corp_name_key and corp_name_key in query_key:
                score += 6000

            score += self.tool_service._score_query_match(corp_name_query, corp_name) * 3
            score += self.tool_service._score_query_match(corp_name_query, row.get("corp_eng_name"))
            score += self.tool_service._score_query_match(corp_name_query, stock_code) * 6
            if score <= 0:
                continue

            ranked.append(
                (
                    score,
                    {
                        "corp_name": row.get("corp_name"),
                        "corp_code": row.get("corp_code"),
                        "stock_code": stock_code,
                        "corp_eng_name": row.get("corp_eng_name"),
                    },
                )
            )

        ranked.sort(
            key=lambda item: (
                -item[0],
                len(str(item[1].get("corp_name") or "")),
                str(item[1].get("corp_name") or ""),
            )
        )
        return [candidate for _, candidate in ranked[:limit]]

    def _resolve_company_from_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        exact_candidates = self._collect_exact_company_candidates(plan)
        if exact_candidates:
            picked = exact_candidates[0]
            self._log(
                "Resolved company by exact candidates",
                {
                    "corp_name_query": plan.get("corp_name_query"),
                    "candidate_count": len(exact_candidates),
                    "candidates": exact_candidates[:5],
                    "resolved": picked,
                },
                kind="state",
            )
            return picked

        attempts = self._candidate_company_attempts(plan, max_attempts=10)
        if not attempts:
            raise ValueError("Could not determine company name from the request.")

        attempted_queries: List[str] = []
        for index, corp_name_query in enumerate(attempts, start=1):
            attempted_queries.append(corp_name_query)
            self._log(
                "Company resolution attempt",
                {
                    "attempt": index,
                    "max_attempts": len(attempts),
                    "corp_name_query": corp_name_query,
                },
                kind="state",
            )
            result = self.execute_tool("search_company", {"corp_name_query": corp_name_query, "limit": 5})
            if not self._result_ok(result) or not result.get("candidates"):
                fallback_candidates = self._search_company_by_normalized_key(corp_name_query, limit=5)
                if not fallback_candidates:
                    continue
                enriched_candidates = []
                for candidate in fallback_candidates:
                    enriched = dict(candidate)
                    enriched["corp_cls"] = self._load_corp_cls(candidate.get("corp_code"))
                    enriched_candidates.append(enriched)
                picked = self._pick_company_candidate(corp_name_query, enriched_candidates) or enriched_candidates[0]
            else:
                enriched_candidates = []
                for candidate in result["candidates"]:
                    enriched = dict(candidate)
                    enriched["corp_cls"] = self._load_corp_cls(candidate.get("corp_code"))
                    enriched_candidates.append(enriched)
                picked = self._pick_company_candidate(corp_name_query, enriched_candidates) or enriched_candidates[0]
            self._log(
                "Resolved company",
                {
                    "attempt": index,
                    "query": corp_name_query,
                    "resolved": picked,
                },
                kind="state",
            )
            return picked

        raise ValueError(
            "Could not resolve company for plan candidates: "
            + ", ".join(attempted_queries[:10])
        )

    def _plan_report_request(self, user_text: str) -> List[Dict[str, Any]]:
        fallback_business_year = self._infer_business_year_fallback(user_text)
        prompt = (
            "You are selecting one or more DART filing plans.\n"
            "Identify which company or companies should be searched, using the company names as they are typically written in DART disclosures.\n"
            "For each plan, provide up to 3 company-name candidates ordered by likelihood of matching the official DART company name.\n"
            "If the user used an abbreviated brand-style or English-letter style such as SK, LG, CJ, KT, or similar, include a likely official Korean DART-style spelling candidate first when possible.\n"
            "Return multiple plans only if needed. Use multiple plans when the user request requires multiple companies, multiple years, or multiple filings such as consecutive quarterly reports.\n"
            "Each plan should represent one filing search target.\n"
            "Do not create duplicate or overlapping plans unless they are clearly required.\n"
            "A plan must also include request_mode. Allowed values are: section_text, filing_list, filing_existence.\n"
            "Use section_text when the user wants document body passages, footnotes, tables, or filing text itself.\n"
            "Use filing_list when the user wants recent filings, filing titles, filing history, or what was filed on a date or during a period.\n"
            "Use filing_existence when the user mainly wants to know whether a filing exists on a specific date or in a period.\n"
            "For filing_list or filing_existence requests, do not force A001 or another specific document-body plan unless the user explicitly asks for that filing body.\n"
            "In filing_list or filing_existence cases, pblntf_ty, pblntf_detail_ty, reprt_code, and report_name_query may be null.\n"
            f"{self._filing_family_guidance_text()}\n"
            f"{self._planning_time_context_text()}\n"
            "Do not invent a guessed absolute date range when the user used a relative period such as recent 6 months or recent 1 year. Preserve the relative period meaning and rely on the runtime date context above.\n"
            "Use the code tables below.\n"
            "Return JSON only. Preferred schema:\n"
            '{"plans":[{"corp_name_query":"SKC","corp_name_candidates":["에스케이씨","SKC","SK 씨"],"plan_goal":"2025 annual depreciation","business_year":2025,"pblntf_ty":"A","pblntf_detail_ty":"A001","reprt_code":"11011","fs_div":"CFS","report_name_query":"","request_mode":"section_text","reason":"..."}]}\n'
            "If only one plan is needed, you may still return it inside plans.\n\n"
            f"User request:\n{self._request_text_with_runtime_date(user_text)}\n\n"
            f"{self._report_code_reference_text()}"
        )
        payload = self._ask_json(prompt, phase="report-plan")
        raw_plans = payload.get("plans") if isinstance(payload.get("plans"), list) else [payload]
        normalized_plans: List[Dict[str, Any]] = []
        for raw_plan in raw_plans:
            if not isinstance(raw_plan, dict):
                continue
            corp_name_query = str(raw_plan.get("corp_name_query") or "").strip()
            if not corp_name_query:
                continue
            raw_candidates = raw_plan.get("corp_name_candidates") or []
            corp_name_candidates: List[str] = []
            if isinstance(raw_candidates, list):
                for candidate in raw_candidates:
                    candidate_text = str(candidate or "").strip()
                    if candidate_text and candidate_text not in corp_name_candidates:
                        corp_name_candidates.append(candidate_text)
            if corp_name_query not in corp_name_candidates:
                corp_name_candidates.insert(0, corp_name_query)
            corp_name_candidates = corp_name_candidates[:3]
            request_mode = self._normalize_request_mode(user_text, raw_plan.get("request_mode"))
            business_year = self._normalize_relative_business_year(
                user_text,
                raw_plan.get("business_year") or fallback_business_year,
            )
            pblntf_detail_ty_raw = str(raw_plan.get("pblntf_detail_ty") or "").strip().upper()
            pblntf_detail_ty = pblntf_detail_ty_raw or None
            pblntf_ty = str(
                raw_plan.get("pblntf_ty") or self._infer_pblntf_ty(pblntf_detail_ty) or ""
            ).strip().upper() or None
            reprt_code = str(raw_plan.get("reprt_code") or "").strip() or None
            fs_div = str(raw_plan.get("fs_div") or "CFS").upper()
            if fs_div not in {"CFS", "OFS"}:
                fs_div = "CFS"
            report_name_query = self._normalize_report_name_query(
                pblntf_detail_ty,
                pblntf_ty,
                raw_plan.get("report_name_query"),
            )
            normalized_plans.append(
                {
                    "corp_name_query": corp_name_query,
                    "corp_name_candidates": corp_name_candidates,
                    "plan_goal": str(raw_plan.get("plan_goal") or "").strip() or user_text,
                    "business_year": business_year,
                    "pblntf_ty": pblntf_ty,
                    "pblntf_detail_ty": pblntf_detail_ty,
                    "reprt_code": reprt_code,
                    "fs_div": fs_div,
                    "report_name_query": report_name_query,
                    "request_mode": request_mode,
                    "reason": str(raw_plan.get("reason") or "").strip(),
                }
            )
        normalized_plans = self._dedupe_plans(normalized_plans)
        if not normalized_plans:
            raise ValueError("LLM did not return any usable filing plan.")
        self._log("Report plans", normalized_plans, kind="state")
        return normalized_plans

    @staticmethod
    def _sort_reports_for_plan(
        reports: List[Dict[str, Any]],
        plan: Dict[str, Any],
        user_text: str = "",
    ) -> List[Dict[str, Any]]:
        target_year = str(plan["business_year"])
        target_reprt = str(plan.get("reprt_code") or "")
        report_name_query = str(plan.get("report_name_query") or "")
        target_detail = str(plan.get("pblntf_detail_ty") or "")
        keyword_source = " ".join(
            str(value or "")
            for value in [user_text, plan.get("plan_goal"), plan.get("reason"), report_name_query]
        )
        keywords = [token for token in re.split(r"[^0-9a-zA-Z가-힣]+", keyword_source.lower()) if len(token) >= 2]

        def score(report: Dict[str, Any]) -> Tuple[int, str]:
            report_nm = str(report.get("report_nm") or "")
            report_nm_lower = report_nm.lower()
            points = 0
            if str(report.get("business_year") or "") == target_year:
                points += 500
            if target_year and target_year in report_nm:
                points += 120
            if target_reprt and str(report.get("reprt_code") or "") == target_reprt:
                points += 400
            if report_name_query and report_name_query in report_nm:
                points += 150
            if target_detail and str(report.get("pblntf_detail_ty") or "") == target_detail:
                points += 300
            for keyword in keywords:
                if keyword in report_nm_lower:
                    points += 180 + min(len(keyword), 12)
            points += len(report_nm)
            return points, str(report.get("rcept_dt") or "")

        return sorted(reports, key=lambda item: (score(item)[0], score(item)[1]), reverse=True)

    @staticmethod
    def _sort_reports_by_latest(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            reports,
            key=lambda item: (
                str(item.get("rcept_dt") or ""),
                str(item.get("report_nm") or ""),
            ),
            reverse=True,
        )

    @staticmethod
    def _audit_detail_order(plan: Dict[str, Any]) -> List[str]:
        if str(plan.get("fs_div") or "CFS").upper() == "CFS":
            return ["F002", "F001", "F003"]
        return ["F001", "F002", "F003"]

    def _with_report_detail(
        self,
        plan: Dict[str, Any],
        detail_code: str,
        *,
        filing_profile: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        adapted = dict(plan)
        adapted["pblntf_detail_ty"] = detail_code
        adapted["pblntf_ty"] = self._infer_pblntf_ty(detail_code)
        adapted["reprt_code"] = None if detail_code.startswith("F") else adapted.get("reprt_code")
        adapted["report_name_query"] = self._normalize_report_name_query(
            detail_code,
            adapted.get("pblntf_ty"),
            adapted.get("report_name_query"),
        )
        if filing_profile:
            adapted["filing_profile"] = filing_profile
        if reason:
            adapted["fallback_reason"] = reason
        return adapted

    @staticmethod
    def _plan_explicitly_requests_audit(plan: Dict[str, Any]) -> bool:
        text = " ".join(
            str(plan.get(key) or "")
            for key in ("plan_goal", "report_name_query", "reason")
        ).lower()
        audit_markers = (
            "감사", "감사의견", "감사보고서", "연결감사보고서", "결합감사보고서",
            "audit", "auditor", "audit report", "audit opinion",
        )
        return any(marker in text for marker in audit_markers)

    def _adapt_plan_to_filing_profile(self, corp: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        adapted = dict(plan)
        corp_cls = str(corp.get("corp_cls") or "").strip().upper()
        adapted["corp_cls"] = corp_cls

        original_detail = str(plan.get("pblntf_detail_ty") or "").upper()
        if corp_cls in {"Y", "K"}:
            if original_detail in {"F001", "F002", "F003"} and not self._plan_explicitly_requests_audit(plan):
                adapted = self._with_report_detail(
                    adapted,
                    "A001",
                    filing_profile="LISTED_REGULAR",
                    reason="listed-prefer-annual-over-audit",
                )
                self._log(
                    "Plan adapted",
                    {
                        "reason": "listed-prefer-annual-over-audit",
                        "from": original_detail,
                        "to": adapted["pblntf_detail_ty"],
                        "corp_name": corp.get("corp_name"),
                    },
                    kind="state",
                )
                return adapted
            adapted["filing_profile"] = "LISTED_REGULAR"
            return adapted

        if corp_cls == "N":
            adapted["filing_profile"] = "KONEX"
            if original_detail in {"A002", "A003"}:
                adapted = self._with_report_detail(
                    adapted,
                    "A001",
                    filing_profile="KONEX",
                    reason="konex-use-annual-first",
                )
                self._log(
                    "Plan adapted",
                    {
                        "reason": "konex-use-annual-first",
                        "from": original_detail,
                        "to": adapted["pblntf_detail_ty"],
                        "corp_name": corp.get("corp_name"),
                    },
                    kind="state",
                )
            return adapted

        if corp_cls == "E" and original_detail in {"A001", "A002", "A003"}:
            detail_code = self._audit_detail_order(plan)[0]
            adapted = self._with_report_detail(
                adapted,
                detail_code,
                filing_profile="OTHER_CORP_AUDIT",
                reason="other-corp-use-audit-first",
            )
            self._log(
                "Plan adapted",
                {
                    "reason": "other-corp-use-audit-first",
                    "from": original_detail,
                    "to": adapted["pblntf_detail_ty"],
                    "corp_name": corp.get("corp_name"),
                },
                kind="state",
            )
            return adapted

        adapted["filing_profile"] = "UNKNOWN"
        return adapted

    def _report_resolution_candidates(self, corp: Dict[str, Any], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        base_plan = self._adapt_plan_to_filing_profile(corp, plan)
        candidates: List[Dict[str, Any]] = []
        seen = set()

        def add(candidate: Dict[str, Any]) -> None:
            key = (
                str(candidate.get("business_year") or ""),
                str(candidate.get("pblntf_ty") or ""),
                str(candidate.get("pblntf_detail_ty") or ""),
                str(candidate.get("reprt_code") or ""),
                str(candidate.get("report_name_query") or ""),
            )
            if key in seen:
                return
            seen.add(key)
            candidates.append(candidate)

        add(base_plan)
        for pblntf_ty, detail_code, report_name_query in self._topic_resolution_candidates(
            str(plan.get("plan_goal") or ""),
            base_plan,
        ):
            candidate = dict(base_plan)
            if detail_code:
                candidate["pblntf_detail_ty"] = detail_code
            if pblntf_ty:
                candidate["pblntf_ty"] = pblntf_ty
            if report_name_query and not candidate.get("report_name_query"):
                candidate["report_name_query"] = report_name_query
            add(candidate)

        corp_cls = str(corp.get("corp_cls") or "").strip().upper()
        original_detail = str(base_plan.get("pblntf_detail_ty") or "").upper()
        if original_detail in {"A001", "A002", "A003"}:
            for detail_code in self._audit_detail_order(base_plan):
                add(
                    self._with_report_detail(
                        base_plan,
                        detail_code,
                        filing_profile=str(base_plan.get("filing_profile") or ""),
                        reason=f"{corp_cls or 'UNKNOWN'}-fallback-to-audit",
                    )
                )

        return candidates

    def _fallback_resolution_candidates(self, corp: Dict[str, Any], plan: Dict[str, Any], user_text: str) -> List[Dict[str, Any]]:
        base_plan = self._adapt_plan_to_filing_profile(corp, plan)
        candidates: List[Dict[str, Any]] = []
        seen = set()

        def add(candidate: Dict[str, Any]) -> None:
            key = (
                str(candidate.get("business_year") or ""),
                str(candidate.get("pblntf_ty") or ""),
                str(candidate.get("pblntf_detail_ty") or ""),
                str(candidate.get("reprt_code") or ""),
                str(candidate.get("report_name_query") or ""),
            )
            if key in seen:
                return
            seen.add(key)
            candidates.append(candidate)

        topic_candidates = self._topic_resolution_candidates(user_text, base_plan)
        if topic_candidates:
            for pblntf_ty, detail_code, report_name_query in topic_candidates:
                candidate = dict(base_plan)
                if detail_code:
                    candidate["pblntf_detail_ty"] = detail_code
                if pblntf_ty:
                    candidate["pblntf_ty"] = pblntf_ty
                if report_name_query and not candidate.get("report_name_query"):
                    candidate["report_name_query"] = report_name_query
                add(candidate)
        else:
            detail = str(base_plan.get("pblntf_detail_ty") or "").upper()
            family = str(base_plan.get("pblntf_ty") or self._infer_pblntf_ty(detail) or "").upper()
            corp_cls = str(corp.get("corp_cls") or "").strip().upper()
            if family == "A":
                for detail_code in self._audit_detail_order(base_plan):
                    add(
                        self._with_report_detail(
                            base_plan,
                            detail_code,
                            filing_profile=str(base_plan.get("filing_profile") or ""),
                            reason=f"{corp_cls or 'UNKNOWN'}-fallback-to-audit",
                        )
                    )
            elif family in {"B", "E"}:
                for pblntf_ty, detail_code, report_name_query in [
                    ("B", "B001", base_plan.get("report_name_query")),
                    ("E", "E001", base_plan.get("report_name_query")),
                    ("A", "A001", None),
                ]:
                    candidate = dict(base_plan)
                    candidate["pblntf_ty"] = pblntf_ty
                    candidate["pblntf_detail_ty"] = detail_code
                    if report_name_query:
                        candidate["report_name_query"] = report_name_query
                    add(candidate)
            elif family == "D":
                for pblntf_ty, detail_code in [("D", "D001"), ("D", "D002")]:
                    candidate = dict(base_plan)
                    candidate["pblntf_ty"] = pblntf_ty
                    candidate["pblntf_detail_ty"] = detail_code
                    add(candidate)
            elif family == "F":
                for detail_code in self._audit_detail_order(base_plan):
                    add(
                        self._with_report_detail(
                            base_plan,
                            detail_code,
                            filing_profile=str(base_plan.get("filing_profile") or ""),
                            reason=f"{corp_cls or 'UNKNOWN'}-fallback-within-audit",
                        )
                    )
                add(self._with_report_detail(base_plan, "A001", filing_profile=str(base_plan.get("filing_profile") or ""), reason="audit-fallback-to-annual"))

        for candidate in self._report_resolution_candidates(corp, base_plan):
            add(candidate)
        return candidates

    def _resolve_target_report(self, corp: Dict[str, Any], plan: Dict[str, Any], user_text: str = "") -> Dict[str, Any]:
        candidate_plans = self._fallback_resolution_candidates(corp, plan, user_text)
        for candidate_plan in candidate_plans:
            cache_key = (
                str(corp.get("corp_code") or ""),
                str(candidate_plan.get("business_year") or ""),
                str(candidate_plan.get("pblntf_ty") or ""),
                str(candidate_plan.get("pblntf_detail_ty") or ""),
                str(candidate_plan.get("reprt_code") or ""),
                str(candidate_plan.get("fs_div") or ""),
                str(candidate_plan.get("report_name_query") or ""),
            )
            cached = self._report_resolution_cache.get(cache_key)
            if cached:
                return dict(cached)

            year = int(candidate_plan["business_year"])
            normalized_last_reprt_at = self.tool_service._default_last_reprt_at(
                candidate_plan.get("pblntf_detail_ty"),
                candidate_plan.get("pblntf_ty"),
            )
            normalized_report_name_query = self.tool_service._normalize_report_name_query(
                candidate_plan.get("pblntf_detail_ty"),
                candidate_plan.get("pblntf_ty"),
                candidate_plan.get("report_name_query"),
            )
            search_result = self.execute_tool(
                "search_reports",
                {
                    "corp_name": corp["corp_name"],
                    "bgn_de": f"{year}0101",
                    "end_de": f"{year + 1}1231",
                    "pblntf_ty": candidate_plan.get("pblntf_ty"),
                    "pblntf_detail_ty": candidate_plan["pblntf_detail_ty"],
                    "last_reprt_at": normalized_last_reprt_at,
                    "report_name_query": normalized_report_name_query,
                    "limit": 30,
                },
            )
            reports = search_result.get("reports", []) if self._result_ok(search_result) else []
            if candidate_plan.get("reprt_code"):
                reports = [
                    report for report in reports
                    if str(report.get("reprt_code") or "") == str(candidate_plan["reprt_code"])
                ] or reports
            reports = self._sort_reports_for_plan(reports, candidate_plan, user_text=user_text)
            if reports:
                chosen = dict(reports[0])
                chosen["resolved_via_plan"] = dict(candidate_plan)
                self._report_resolution_cache[cache_key] = dict(chosen)
                self._log("Resolved report", chosen, kind="state")
                return chosen

            if candidate_plan["pblntf_detail_ty"] == "A001":
                fallback = self.execute_tool(
                    "find_business_report",
                    {
                        "corp_name": corp["corp_name"],
                        "report_year": year,
                        "prefer_final_report": True,
                    },
                )
                if self._result_ok(fallback) and fallback.get("report"):
                    chosen = dict(fallback["report"])
                    chosen["resolved_via_plan"] = dict(candidate_plan)
                    self._report_resolution_cache[cache_key] = dict(chosen)
                    self._log("Resolved report", chosen, kind="state")
                    return chosen

        fallback = self._resolve_report_by_recent_filings(corp, plan, user_text)
        if fallback:
            return fallback

        final_plan = candidate_plans[-1] if candidate_plans else self._adapt_plan_to_filing_profile(corp, plan)
        raise ValueError(
            f"Could not resolve a filing for {corp['corp_name']} / year={final_plan.get('business_year')} / code={final_plan.get('pblntf_detail_ty')} / profile={final_plan.get('filing_profile')}"
        )

    @classmethod
    def _recent_filing_window(cls, user_text: str, plan: Dict[str, Any]) -> Tuple[str, str]:
        explicit = cls._extract_explicit_date_window(user_text)
        if explicit:
            return explicit
        relative = cls._extract_relative_period_window(user_text)
        if relative:
            return relative
        year = int(plan.get("business_year") or cls._infer_business_year_fallback(user_text))
        return f"{year}0101", f"{year + 1}1231"

    @staticmethod
    def _filing_row_source_path(corp: Dict[str, Any], report: Dict[str, Any]) -> str:
        corp_name = str(corp.get("corp_name") or "").strip()
        rcept_dt = str(report.get("rcept_dt") or "").strip()
        report_nm = OpenDartReportExplorer._normalize_text(str(report.get("report_nm") or ""))
        return "_".join(part for part in [corp_name, rcept_dt, report_nm] if part)

    @classmethod
    def _filing_row_text(cls, corp: Dict[str, Any], reports: List[Dict[str, Any]], heading: str) -> str:
        lines = [heading]
        for index, report in enumerate(reports, start=1):
            lines.append(
                f"{index}. {report.get('rcept_dt')} | {OpenDartReportExplorer._normalize_text(str(report.get('report_nm') or ''))} | 접수번호 {report.get('rcept_no')}"
            )
        return "\n".join(line for line in lines if line).strip()

    def _score_recent_filing_candidate(self, report: Dict[str, Any], plan: Dict[str, Any], user_text: str) -> int:
        report_nm = str(report.get("report_nm") or "")
        report_nm_lower = report_nm.lower()
        score = 0
        if plan.get("report_name_query") and str(plan["report_name_query"]).lower() in report_nm_lower:
            score += 400
        if str(plan.get("pblntf_detail_ty") or "") and str(plan.get("pblntf_detail_ty") or "") == str(report.get("pblntf_detail_ty") or ""):
            score += 250
        for keyword in re.split(r"[^0-9a-zA-Z가-힣]+", f"{user_text} {plan.get('plan_goal')} {plan.get('reason')}"):
            kw = keyword.lower().strip()
            if len(kw) >= 2 and kw in report_nm_lower:
                score += 80 + min(len(kw), 12)
        score += len(report_nm)
        return score

    @staticmethod
    def _recent_filing_query_keywords(plan: Dict[str, Any]) -> List[str]:
        query = str(plan.get("report_name_query") or "").strip().lower()
        if not query:
            return []
        stopwords = {
            "보고서",
            "공시",
            "최근",
            "최신",
            "자료",
            "조회",
            "목록",
            "내역",
            "또는",
            "and",
            "or",
        }
        keywords: List[str] = []
        for token in re.split(r"[^0-9a-zA-Z가-힣]+", query):
            word = token.strip()
            if len(word) < 2 or word in stopwords:
                continue
            if word not in keywords:
                keywords.append(word)
        return keywords

    def _is_strong_recent_filing_match(self, report: Dict[str, Any], plan: Dict[str, Any]) -> bool:
        query = str(plan.get("report_name_query") or "").strip().lower()
        if not query:
            return True
        report_nm = str(report.get("report_nm") or "").strip().lower()
        compact_query = re.sub(r"\s+", "", query)
        compact_report_nm = re.sub(r"\s+", "", report_nm)
        if compact_query and compact_query in compact_report_nm:
            return True
        keywords = self._recent_filing_query_keywords(plan)
        if not keywords:
            return True
        matched = [keyword for keyword in keywords if keyword in report_nm]
        return len(matched) >= 2 or any(len(keyword) >= 4 for keyword in matched)

    def _collect_recent_filing_materials(
        self,
        corp: Dict[str, Any],
        plan: Dict[str, Any],
        user_text: str,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        stock_code = str(corp.get("stock_code") or "").strip()
        if not stock_code:
            return [], []
        bgn_de, end_de = self._recent_filing_window(user_text, plan)
        result = self.execute_tool(
            "search_recent_filings_by_stock_code",
            {
                "stock_code": stock_code,
                "bgn_de": bgn_de,
                "end_de": end_de,
                "pblntf_ty": plan.get("pblntf_ty"),
                "pblntf_detail_ty": plan.get("pblntf_detail_ty"),
                "last_reprt_at": "Y",
                "limit": 50,
            },
        )
        reports = result.get("reports", []) if self._result_ok(result) else []
        if not reports:
            return [], []

        reports = sorted(
            reports,
            key=lambda item: (
                self._score_recent_filing_candidate(item, plan, user_text),
                str(item.get("rcept_dt") or ""),
            ),
            reverse=True,
        )
        strong_matches = [report for report in reports if self._is_strong_recent_filing_match(report, plan)]
        if strong_matches:
            reports = strong_matches
        request_mode = str(plan.get("request_mode") or "section_text")
        if request_mode == "filing_existence":
            top = reports[:1]
            text = self._filing_row_text(
                corp,
                top,
                heading=f"{corp.get('corp_name')} 공시 존재 확인 결과",
            )
            return ([{"text": text, "source_path": self._filing_row_source_path(corp, top[0])}], top)

        selected = reports[: min(len(reports), 10)]
        materials: List[Dict[str, str]] = []
        for index, report in enumerate(selected, start=1):
            heading = f"{corp.get('corp_name')} 공시 목록" if index == 1 else f"{corp.get('corp_name')} 공시"
            text = self._filing_row_text(corp, [report], heading=heading)
            materials.append(
                {
                    "text": text,
                    "source_path": self._filing_row_source_path(corp, report),
                }
            )
        return materials, selected

    def _resolve_report_by_recent_filings(
        self,
        corp: Dict[str, Any],
        plan: Dict[str, Any],
        user_text: str,
    ) -> Optional[Dict[str, Any]]:
        stock_code = str(corp.get("stock_code") or "").strip()
        if not stock_code:
            return None
        bgn_de, end_de = self._recent_filing_window(user_text, plan)
        result = self.execute_tool(
            "search_recent_filings_by_stock_code",
            {
                "stock_code": stock_code,
                "bgn_de": bgn_de,
                "end_de": end_de,
                "pblntf_ty": plan.get("pblntf_ty"),
                "pblntf_detail_ty": plan.get("pblntf_detail_ty"),
                "last_reprt_at": "Y",
                "limit": 30,
            },
        )
        reports = result.get("reports", []) if self._result_ok(result) else []
        if not reports:
            return None
        reports = sorted(
            reports,
            key=lambda item: (
                self._score_recent_filing_candidate(item, plan, user_text),
                str(item.get("rcept_dt") or ""),
            ),
            reverse=True,
        )
        strong_matches = [report for report in reports if self._is_strong_recent_filing_match(report, plan)]
        if strong_matches:
            reports = strong_matches
        chosen = dict(reports[0])
        chosen["resolved_via_plan"] = dict(plan)
        self._log("Resolved report by recent filings fallback", chosen, kind="state")
        return chosen

    @staticmethod
    def _failure_label(failure_type: str) -> str:
        return FAILURE_TYPE_LABELS.get(str(failure_type or "").strip(), FAILURE_TYPE_LABELS["unknown_failure"])

    def _format_failure_summary(self, failure_type: str, failure_reason: str) -> str:
        label = self._failure_label(failure_type)
        reason = str(failure_reason or "").strip()
        return f"{label}: {reason}" if reason else label

    @staticmethod
    def _classify_exception_failure(message: str) -> str:
        lowered = str(message or "").lower()
        if "resolve company" in lowered or "회사" in lowered and "찾" in lowered:
            return "company_identification_failed"
        if "resolve a filing" in lowered or "보고서" in lowered:
            return "report_not_found"
        return "unknown_failure"

    @staticmethod
    def _classify_plan_outcome(
        *,
        request_mode: str,
        materials: List[Dict[str, str]],
        evidence_items: List[Dict[str, Any]],
        sufficiency_ok: bool,
    ) -> Tuple[str, str]:
        if materials:
            return "", ""
        if request_mode in {"filing_list", "filing_existence"}:
            return "recent_filings_fallback_failed", "최근 공시 목록까지 확인했지만 요청에 맞는 공시 근거를 찾지 못했습니다."
        if evidence_items and not sufficiency_ok:
            return "insufficient_evidence", "관련 위치를 확인했지만 요청을 충분히 뒷받침할 근거가 부족했습니다."
        return "section_not_found", "선택된 보고서에서 요청과 직접 관련된 섹션을 찾지 못했습니다."

    @classmethod
    def _group_material_header(cls, source_path: str) -> str:
        value = str(source_path or "").strip()
        if not value:
            return "[Material]"
        parts = [part.strip() for part in value.split("_") if part.strip()]
        return f"[{parts[0]}]" if parts else "[Material]"

    @classmethod
    def _merge_materials(cls, materials: List[Dict[str, str]]) -> Dict[str, Any]:
        merged_sections: List[str] = []
        source_paths: List[str] = []
        seen_text_source = set()
        seen_paths = set()
        kept_materials: List[Dict[str, str]] = []

        for material in materials:
            text = str(material.get("text") or "").strip()
            source_path = str(material.get("source_path") or "").strip()
            if not text:
                continue
            key = (text, source_path)
            if key in seen_text_source:
                continue
            seen_text_source.add(key)
            kept_materials.append({"text": text, "source_path": source_path})
            header = cls._group_material_header(source_path)
            body = text
            if source_path:
                body = f"{header}\n[source] {source_path}\n{body}"
            else:
                body = f"{header}\n{body}"
            merged_sections.append(body)
            if source_path and source_path not in seen_paths:
                seen_paths.add(source_path)
                source_paths.append(source_path)

        return {
            "text": "\n\n".join(merged_sections).strip(),
            "source_paths": source_paths,
            "materials": kept_materials,
        }

    def _load_report_toc_catalog(self, report: Dict[str, Any]) -> Dict[str, Any]:
        cached = self._toc_catalog_cache.get(str(report["rcept_no"]))
        if cached:
            return cached
        xml_entries = self.tool_service.explorer.list_document_toc_entries(report["rcept_no"])
        toc_entries: List[Dict[str, Any]] = []
        if xml_entries:
            for index, entry in enumerate(xml_entries, start=1):
                toc_entries.append(
                    {
                        "toc_ref": f"T{index:03d}",
                        "page_title": entry.get("parent_title") or entry.get("doc_title") or entry.get("section_title"),
                        "section_title": entry.get("section_title"),
                        "toc_id": entry.get("toc_id"),
                        "whole_page": False,
                        "source_type": "document_xml",
                        "xml_member": entry.get("xml_member"),
                        "level": entry.get("level"),
                        "doc_title": entry.get("doc_title"),
                        "parent_title": entry.get("parent_title"),
                    }
                )
            catalog = {
                "report": report,
                "pages": [],
                "entries": toc_entries,
                "source": "document_xml",
            }
            self._log(
                "TOC catalog built",
                {"source": "document_xml", "toc_entry_count": len(catalog["entries"])},
                kind="state",
            )
            self._toc_catalog_cache[str(report["rcept_no"])] = catalog
            return catalog

        pages_result = self.execute_tool("list_report_pages", {"report_id": report["report_id"]})
        if not self._result_ok(pages_result):
            raise ValueError(f"Could not list report pages: {pages_result.get('error')}")

        toc_index = 1
        for page in pages_result.get("pages", []):
            time.sleep(random.uniform(0.2, 0.5))
            subsection_result = self.execute_tool("list_page_subsections", {"page_id": page["page_id"]})
            sections = subsection_result.get("sections", []) if self._result_ok(subsection_result) else []
            if sections:
                for section in sections:
                    toc_entries.append(
                        {
                            "toc_ref": f"T{toc_index:03d}",
                            "page_id": page["page_id"],
                            "page_title": page["title"],
                            "toc_id": section.get("toc_id"),
                            "section_title": section.get("title"),
                            "whole_page": False,
                            "source_type": "viewer_html",
                            "xml_member": None,
                            "level": None,
                            "doc_title": page["title"],
                            "parent_title": None,
                        }
                    )
                    toc_index += 1
                continue

            toc_entries.append(
                {
                    "toc_ref": f"T{toc_index:03d}",
                    "page_id": page["page_id"],
                    "page_title": page["title"],
                    "toc_id": None,
                    "section_title": page["title"],
                    "whole_page": True,
                    "source_type": "viewer_html",
                    "xml_member": None,
                    "level": None,
                    "doc_title": page["title"],
                    "parent_title": None,
                }
            )
            toc_index += 1

        catalog = {
            "report": report,
            "pages": pages_result.get("pages", []),
            "entries": toc_entries,
            "source": "viewer_html",
        }
        self._log(
            "TOC catalog built",
            {
                "source": "viewer_html",
                "page_count": len(catalog["pages"]),
                "toc_entry_count": len(catalog["entries"]),
            },
            kind="state",
        )
        self._toc_catalog_cache[str(report["rcept_no"])] = catalog
        return catalog

    @staticmethod
    def _format_toc_catalog(entries: List[Dict[str, Any]], excluded_refs: Optional[List[str]] = None) -> str:
        excluded = set(excluded_refs or [])
        lines: List[str] = []
        for entry in entries:
            if entry["toc_ref"] in excluded:
                continue
            toc_id = entry["toc_id"] or "WHOLE_PAGE"
            parent = entry.get("parent_title") or entry.get("page_title") or ""
            lines.append(
                f'{entry["toc_ref"]} | parent="{parent}" | toc_id="{toc_id}" | section="{entry["section_title"]}"'
            )
        return "\n".join(lines)

    def _fallback_toc_refs(
        self,
        user_text: str,
        entries: List[Dict[str, Any]],
        excluded_refs: List[str],
        limit: int = 3,
    ) -> List[str]:
        excluded = set(excluded_refs)
        ranked = []
        for entry in entries:
            if entry["toc_ref"] in excluded:
                continue
            score = self.tool_service._score_query_match(
                user_text,
                f'{entry["page_title"]} {entry["section_title"]}',
            )
            if score > 0:
                ranked.append((score, entry["toc_ref"]))
        ranked.sort(reverse=True)
        refs = [toc_ref for _, toc_ref in ranked[:limit]]
        if refs:
            return refs
        return [entry["toc_ref"] for entry in entries if entry["toc_ref"] not in excluded][:limit]

    def _normalize_toc_refs(
        self,
        payload: Dict[str, Any],
        entries: List[Dict[str, Any]],
        excluded_refs: List[str],
        fallback_question: str,
    ) -> List[str]:
        allowed = {entry["toc_ref"] for entry in entries if entry["toc_ref"] not in set(excluded_refs)}
        refs = payload.get("toc_refs") or payload.get("candidate_toc_refs") or payload.get("candidates") or []
        normalized: List[str] = []
        for ref in refs:
            ref_text = str(ref).strip().upper()
            if ref_text in allowed and ref_text not in normalized:
                normalized.append(ref_text)
        if normalized:
            return normalized
        return self._fallback_toc_refs(fallback_question, entries, excluded_refs)

    def _render_blocks(self, blocks: List[Dict[str, Any]], limit: Optional[int] = None) -> str:
        parts: List[str] = []
        for block in blocks:
            block_type = block.get("type")
            content = str(block.get("content") or "").strip()
            if not content:
                continue
            if block_type == "table":
                parts.append("[TABLE]\n" + content)
            else:
                parts.append(content)
        return self._compact_text("\n\n".join(parts), limit=limit or self.EVIDENCE_RENDER_LIMIT)

    @classmethod
    def _split_text_chunks(cls, text: str) -> List[str]:
        value = str(text or "").strip()
        if not value:
            return []
        if len(value) <= cls.EVIDENCE_CHUNK_SIZE:
            return [value]

        chunks: List[str] = []
        step = max(1, cls.EVIDENCE_CHUNK_SIZE - cls.EVIDENCE_CHUNK_OVERLAP)
        start = 0
        total = len(value)
        while start < total:
            end = min(total, start + cls.EVIDENCE_CHUNK_SIZE)
            if end < total:
                window = value[start:end]
                split_candidates = [
                    window.rfind("\n\n"),
                    window.rfind("\n"),
                    window.rfind(" | "),
                    window.rfind(". "),
                ]
                best = max(split_candidates)
                if best >= int(len(window) * 0.6):
                    end = start + best + 1
            chunk = value[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= total:
                break
            start = max(start + 1, end - cls.EVIDENCE_CHUNK_OVERLAP)
            if len(chunks) >= 24:
                break
        return chunks

    @staticmethod
    def _append_unique_excerpt(parts: List[str], candidate: str) -> None:
        text = OpenDartReportExplorer._normalize_text(candidate)
        if not text:
            return
        for existing in parts:
            if text == existing or text in existing or existing in text:
                return
        parts.append(text)

    def _focus_single_evidence_item(
        self,
        user_text: str,
        plan: Dict[str, Any],
        corp: Dict[str, Any],
        report: Dict[str, Any],
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        content = str(item.get("content") or "").strip()
        if not content or len(content) <= self.LONG_EVIDENCE_THRESHOLD:
            return item

        section_label = str(item.get("section_title") or item.get("page_title") or "").strip()
        chunks = self._split_text_chunks(content)
        if len(chunks) <= 1:
            return item

        focused_parts: List[str] = []
        for index, chunk in enumerate(chunks, start=1):
            prompt = (
                "You are filtering a long DART filing excerpt for a retrieval tool.\n"
                "Return JSON only with this schema:\n"
                '{"relevant":true,"focused_text":"...","reason":"..."}\n'
                "Decide whether this chunk contains content directly useful for the user request.\n"
                "If relevant, keep the useful part with enough detail to answer later.\n"
                "Preserve note headings, numbers, units, date ranges, conditions, table labels, and accounting terms.\n"
                "Do not invent content. Do not remove important details. Light cleanup is allowed, but keep the source wording and ordering as much as possible.\n"
                "If the chunk is not useful, return relevant=false and focused_text=\"\".\n\n"
                f"User request:\n{self._request_text_with_runtime_date(user_text)}\n\n"
                "Plan context:\n"
                f"{self._plan_scope_summary(plan, corp=corp, report=report)}\n\n"
                f"Current evidence section: {section_label}\n"
                f"Chunk {index} of {len(chunks)}:\n{chunk}"
            )
            payload = self._ask_json(prompt, phase="evidence-focus")
            if bool(payload.get("relevant")):
                self._append_unique_excerpt(focused_parts, str(payload.get("focused_text") or ""))

        if not focused_parts:
            return item

        focused_text = "\n\n".join(focused_parts)
        focused_text = self._compact_text(focused_text, limit=self.FOCUSED_EVIDENCE_LIMIT)
        focused_item = dict(item)
        focused_item["content"] = focused_text
        focused_item["content_focused"] = True
        focused_item["raw_content_length"] = len(content)
        focused_item["focused_content_length"] = len(focused_text)
        self._log(
            "Evidence focused",
            {
                "toc_ref": item.get("toc_ref"),
                "section_title": section_label,
                "raw_length": len(content),
                "focused_length": len(focused_text),
                "focused_chunks": len(focused_parts),
            },
            kind="state",
        )
        return focused_item

    def _prepare_evidence_items_for_review(
        self,
        user_text: str,
        plan: Dict[str, Any],
        corp: Dict[str, Any],
        report: Dict[str, Any],
        evidence_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []
        for item in evidence_items:
            prepared.append(self._focus_single_evidence_item(user_text, plan, corp, report, item))
        return prepared

    def _extract_toc_entries(
        self,
        catalog: Dict[str, Any],
        toc_refs: List[str],
    ) -> List[Dict[str, Any]]:
        entry_map = {entry["toc_ref"]: entry for entry in catalog["entries"]}
        selected_entries = [entry_map[ref] for ref in toc_refs if ref in entry_map]
        evidence_items: List[Dict[str, Any]] = []
        for entry in selected_entries:
            cache_key = (
                str(catalog["report"].get("rcept_no") or ""),
                str(entry.get("source_type") or ""),
                str(entry.get("xml_member") or entry.get("page_id") or ""),
                str(entry.get("toc_id") or ""),
            )
            cached_result = self._evidence_cache.get(cache_key)
            if cached_result is not None:
                result = cached_result
            else:
                result: Dict[str, Any]
                if entry.get("source_type") == "document_xml":
                    try:
                        result = self.tool_service.explorer.extract_document_toc_section(
                            rcept_no=catalog["report"]["rcept_no"],
                            xml_member=entry["xml_member"],
                            toc_id=str(entry["toc_id"]),
                            max_blocks=180,
                        )
                        result["ok"] = True
                    except Exception as e:
                        result = {"ok": False, "error": str(e), "blocks": []}
                else:
                    targets = [
                        {
                            "page_id": entry["page_id"],
                            "keyword": None,
                            "toc_id": entry["toc_id"],
                            "whole_page": bool(entry["whole_page"]),
                        }
                    ]
                    extracted = self.execute_tool(
                        "extract_report_sections",
                        {
                            "targets": targets,
                            "max_table_rows": 80,
                            "max_blocks": 140,
                        },
                    )
                    results = extracted.get("results", []) if self._result_ok(extracted) else []
                    result = results[0] if results else {"ok": False, "blocks": []}
                self._evidence_cache[cache_key] = dict(result)

            evidence_items.append(
                {
                    "toc_ref": entry["toc_ref"],
                    "toc_id": entry["toc_id"],
                    "page_title": entry["page_title"],
                    "parent_title": entry.get("parent_title"),
                    "section_title": entry["section_title"],
                    "ok": bool(result.get("ok")),
                    "content": self._render_blocks(result.get("blocks", [])),
                    "source_type": entry.get("source_type"),
                }
            )
        self._log(
            "Evidence batch",
            {
                "selected_toc_refs": toc_refs,
                "usable_evidence_count": sum(1 for item in evidence_items if item["content"]),
            },
            kind="state",
        )
        return evidence_items

    @staticmethod
    def _format_evidence_bundle(evidence_items: List[Dict[str, Any]]) -> str:
        chunks = []
        for item in evidence_items:
            chunks.append(
                f'[{item["toc_ref"]}] page="{item["page_title"]}" section="{item["section_title"]}" '
                f'toc_id="{item["toc_id"] or "WHOLE_PAGE"}"\n{item["content"]}'
            )
        return "\n\n".join(chunks)

    def _select_initial_toc_refs(
        self,
        user_text: str,
        plan: Dict[str, Any],
        corp: Dict[str, Any],
        report: Dict[str, Any],
        catalog: Dict[str, Any],
    ) -> List[str]:
        scoped_query = self._plan_scope_query(user_text, plan)
        prompt = (
            "You are choosing TOC entries from a DART filing.\n"
            "Pick multiple TOC refs that are most likely to contain evidence for this plan goal.\n"
            "Return JSON only.\n"
            '{"toc_refs":["T001","T014"],"reason":"..."}\n\n'
            f"Original user request:\n{self._request_text_with_runtime_date(user_text)}\n\n"
            f"Plan scope:\n{self._plan_scope_summary(plan, corp=corp, report=report)}\n\n"
            f"Report:\n- report_nm: {report.get('report_nm')}\n- business_year: {report.get('business_year')}\n"
            f"- reprt_code: {report.get('reprt_code')}\n- rcept_no: {report.get('rcept_no')}\n\n"
            "TOC catalog:\n"
            f"{self._format_toc_catalog(catalog['entries'])}"
        )
        payload = self._ask_json(prompt, phase="toc-selection")
        refs = self._normalize_toc_refs(payload, catalog["entries"], [], scoped_query)
        self._log("Selected TOC refs", refs, kind="state")
        return refs

    def _review_evidence(
        self,
        user_text: str,
        plan: Dict[str, Any],
        corp: Dict[str, Any],
        report: Dict[str, Any],
        catalog: Dict[str, Any],
        selected_refs: List[str],
        evidence_items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        scoped_query = self._plan_scope_query(user_text, plan)
        prompt = (
            "Check whether the accumulated DART excerpts are sufficient to answer this specific plan goal.\n"
            "If sufficient, return JSON only: {\"status\":\"OKAY\",\"reason\":\"...\"}\n"
            "If insufficient, return JSON only: "
            "{\"status\":\"NO\",\"toc_refs\":[\"T002\",\"T019\"],\"reason\":\"...\"}\n"
            "When status is NO, choose only refs not already selected.\n\n"
            f"Original user request:\n{self._request_text_with_runtime_date(user_text)}\n\n"
            f"Plan scope:\n{self._plan_scope_summary(plan, corp=corp, report=report)}\n\n"
            f"Report:\n- report_nm: {report.get('report_nm')}\n- business_year: {report.get('business_year')}\n"
            f"- reprt_code: {report.get('reprt_code')}\n- fs_div target: {report.get('fs_div') or 'unknown'}\n\n"
            f"Already selected TOC refs:\n{', '.join(selected_refs) if selected_refs else '(none)'}\n\n"
            "Remaining TOC catalog:\n"
            f"{self._format_toc_catalog(catalog['entries'], excluded_refs=selected_refs)}\n\n"
            "Accumulated evidence:\n"
            f"{self._format_evidence_bundle(evidence_items)}"
        )
        payload = self._ask_json(prompt, phase="evidence-review")
        status = str(payload.get("status") or "").strip().upper()
        result = {"status": "OKAY" if status == "OKAY" else "NO", "reason": str(payload.get("reason") or "").strip()}
        if result["status"] == "NO":
            result["toc_refs"] = self._normalize_toc_refs(payload, catalog["entries"], selected_refs, scoped_query)
        self._log("Evidence review", result, kind="state")
        return result

    @staticmethod
    def _evidence_digest(evidence_items: List[Dict[str, Any]], max_items: int = 4, per_item_limit: int = 1200) -> str:
        chunks: List[str] = []
        for item in evidence_items[:max_items]:
            chunks.append(
                f'[{item["toc_ref"]}] {item["section_title"]}\n{str(item.get("content") or "")[:per_item_limit]}'
            )
        return "\n\n".join(chunks)

    @staticmethod
    def _path_segment(value: Any) -> str:
        text = OpenDartReportExplorer._normalize_text(str(value or ""))
        text = text.replace("_", " ").strip(" _/-")
        return text

    def _report_period_label(self, report: Dict[str, Any]) -> str:
        reprt_code_label = self._path_segment(report.get("reprt_code_label"))
        if reprt_code_label:
            return reprt_code_label

        report_nm = self._path_segment(report.get("report_nm"))
        for label in ("사업보고서", "반기보고서", "분기보고서", "1분기보고서", "3분기보고서"):
            if label in report_nm:
                return label
        return report_nm or "보고서"

    def _build_source_path(
        self,
        corp: Dict[str, Any],
        report: Dict[str, Any],
        evidence_item: Dict[str, Any],
    ) -> str:
        company_name = self._path_segment(corp.get("corp_name")) or "회사미상"
        report_year = self._path_segment(
            report.get("business_year")
            or self.tool_service.explorer.try_extract_business_year(report.get("report_nm"))
            or "연도미상"
        )
        report_period = self._report_period_label(report)
        page_title = self._path_segment(evidence_item.get("page_title"))
        section_title = self._path_segment(evidence_item.get("section_title"))
        parent_title = self._path_segment(evidence_item.get("parent_title"))

        location_parts: List[str] = []
        for part in (parent_title, page_title):
            if part and part not in location_parts:
                location_parts.append(part)
        location = "/".join(location_parts)
        if section_title and section_title not in location_parts:
            location = f"{location}/{section_title}" if location else section_title

        if not location:
            location = page_title or section_title or "항목미상"

        return "_".join(
            part for part in [company_name, report_year, report_period, location] if part
        )

    @staticmethod
    def _normalize_selected_evidence_refs(
        payload: Dict[str, Any],
        evidence_items: List[Dict[str, Any]],
        limit: int = 4,
    ) -> List[str]:
        allowed = {
            str(item.get("toc_ref") or "").strip().upper()
            for item in evidence_items
            if item.get("content")
        }
        refs = (
            payload.get("keep_toc_refs")
            or payload.get("toc_refs")
            or payload.get("selected_toc_refs")
            or []
        )

        normalized: List[str] = []
        for ref in refs:
            ref_text = str(ref or "").strip().upper()
            if ref_text in allowed and ref_text not in normalized:
                normalized.append(ref_text)
            if len(normalized) >= limit:
                break
        return normalized

    def _select_material_evidence_refs(
        self,
        user_text: str,
        plan: Dict[str, Any],
        corp: Dict[str, Any],
        report: Dict[str, Any],
        evidence_items: List[Dict[str, Any]],
        sufficiency_ok: bool,
    ) -> List[str]:
        usable_items = [item for item in evidence_items if item.get("content")]
        if not usable_items:
            return []
        if len(usable_items) == 1:
            return [usable_items[0]["toc_ref"]]

        prompt = (
            "Choose the DART evidence excerpts that should be returned by the public function.\n"
            "Return JSON only with this schema:\n"
            '{"keep_toc_refs":["T001","T014"],"reason":"..."}\n'
            "Choose only the excerpts that are actually needed for the user request.\n"
            "Use the evidence as-is. Do not paraphrase or rewrite the excerpt text.\n"
            "Prefer 1 to 4 refs.\n\n"
            f"User request:\n{self._request_text_with_runtime_date(user_text)}\n\n"
            "Plan context:\n"
            f"{self._plan_scope_summary(plan, corp=corp, report=report)}\n"
            f"- evidence_sufficient: {'yes' if sufficiency_ok else 'no'}\n\n"
            "Evidence candidates:\n"
            f"{self._format_evidence_bundle(usable_items)}"
        )
        payload = self._ask_json(prompt, phase="material-selection")
        selected_refs = self._normalize_selected_evidence_refs(payload, usable_items)
        if selected_refs:
            return selected_refs
        return [item["toc_ref"] for item in usable_items[: min(len(usable_items), 2)]]

    def _compose_plan_materials(
        self,
        user_text: str,
        corp: Dict[str, Any],
        plan: Dict[str, Any],
        report: Dict[str, Any],
        evidence_items: List[Dict[str, Any]],
        sufficiency_ok: bool,
    ) -> List[Dict[str, str]]:
        usable_items = [item for item in evidence_items if item.get("content")]
        if not usable_items:
            return []

        selected_refs = self._select_material_evidence_refs(
            user_text=user_text,
            plan=plan,
            corp=corp,
            report=report,
            evidence_items=usable_items,
            sufficiency_ok=sufficiency_ok,
        )
        selected_ref_set = set(selected_refs)

        materials: List[Dict[str, str]] = []
        seen = set()
        for item in usable_items:
            if item["toc_ref"] not in selected_ref_set:
                continue
            text = OpenDartReportExplorer._normalize_text(str(item.get("content") or ""))
            source_path = self._build_source_path(corp, report, item)
            key = (text, source_path)
            if not text or key in seen:
                continue
            seen.add(key)
            materials.append(
                {
                    "text": text,
                    "source_path": source_path,
                }
            )
        return materials

    def _collect_wrapped_results(self, user_text: str) -> Dict[str, Any]:
        self._log("Provider", getattr(self, "provider_name", self.__class__.__name__), kind="meta")
        self._log("Question", user_text, kind="meta")

        plans = self._plan_report_request(user_text)
        kept_results: List[Dict[str, Any]] = []
        materials: List[Dict[str, str]] = []
        for index, plan in enumerate(plans, start=1):
            kept = self._execute_single_plan(
                user_text,
                plan,
                plan_index=index,
                total_plans=len(plans),
            )
            kept_results.append(kept)
            if kept.get("status") != "ok":
                continue
            materials.extend(kept.get("materials") or [])

        merged = self._merge_materials(materials)
        result = {
            "kept_results": kept_results,
            "materials": merged["materials"],
            "text": merged["text"],
            "source_paths": merged["source_paths"],
        }
        self._log(
            "Collected materials",
            {
                "material_count": len(result["materials"]),
                "source_path_count": len(result["source_paths"]),
            },
            kind="state",
        )
        return result

    def _execute_single_plan(
        self,
        user_text: str,
        plan: Dict[str, Any],
        *,
        plan_index: int,
        total_plans: int,
    ) -> Dict[str, Any]:
        empty_corp = {"corp_name": plan.get("corp_name_query"), "stock_code": None}
        empty_report = {"report_nm": None, "rcept_dt": None}
        request_mode = str(plan.get("request_mode") or "section_text")
        self._log(
            "Plan start",
            {
                "plan_index": plan_index,
                "total_plans": total_plans,
                "corp_name_query": plan.get("corp_name_query"),
                "plan_goal": plan.get("plan_goal"),
                "business_year": plan.get("business_year"),
                "pblntf_detail_ty": plan.get("pblntf_detail_ty"),
                "reprt_code": plan.get("reprt_code"),
                "fs_div": plan.get("fs_div"),
                "request_mode": request_mode,
            },
            kind="state",
        )
        try:
            corp = self._resolve_company_from_plan(plan)
            effective_plan = self._adapt_plan_to_filing_profile(corp, plan)
            accumulated_evidence: List[Dict[str, Any]] = []
            materials: List[Dict[str, str]] = []
            sufficiency_ok = False
            attempts_used = 0
            report = dict(empty_report)

            if request_mode in {"filing_list", "filing_existence"}:
                materials, matched_reports = self._collect_recent_filing_materials(corp, effective_plan, user_text)
                attempts_used = 1
                if matched_reports:
                    report = dict(matched_reports[0])
                    report["fs_div"] = effective_plan["fs_div"]
                failure_type, failure_reason = self._classify_plan_outcome(
                    request_mode=request_mode,
                    materials=materials,
                    evidence_items=[],
                    sufficiency_ok=bool(materials),
                )
                plan_keep = {
                    "what_we_looked_for": str(effective_plan.get("plan_goal") or "").strip(),
                    "confirmed_findings": [material["text"] for material in materials[:1]],
                    "key_numbers": [],
                    "unconfirmed_points": [] if materials else [failure_reason],
                    "plan_result_summary": "공시 목록/존재 여부를 기준으로 정리했습니다." if materials else "공시 목록 기준으로도 충분한 근거를 찾지 못했습니다.",
                    "evidence_digest": "\n\n".join(material["text"] for material in materials[:2]).strip(),
                }
                kept = {
                    "plan_index": plan_index,
                    "status": "ok" if materials else "failed",
                    "corp": corp,
                    "plan": effective_plan,
                    "report": report,
                    "evidence_items": [],
                    "materials": materials,
                    "attempts_used": attempts_used,
                    "sufficiency_ok": bool(materials),
                    "failure_type": failure_type or "",
                    "failure_reason": failure_reason or "",
                }
                self._log("Plan result kept", kept, kind="state" if materials else "error")
                return kept

            report = self._resolve_target_report(corp, effective_plan, user_text=user_text)
            effective_plan = dict(report.get("resolved_via_plan") or effective_plan)
            report["fs_div"] = effective_plan["fs_div"]
            catalog = self._load_report_toc_catalog(report)
            selected_refs = self._select_initial_toc_refs(user_text, effective_plan, corp, report, catalog)
            scoped_query = self._plan_scope_query(user_text, effective_plan)

            for attempt in range(1, 4):
                attempts_used = attempt
                fresh_refs = [ref for ref in selected_refs if ref not in {item["toc_ref"] for item in accumulated_evidence}]
                if not fresh_refs:
                    fresh_refs = self._fallback_toc_refs(
                        scoped_query,
                        catalog["entries"],
                        [item["toc_ref"] for item in accumulated_evidence],
                    )
                if not fresh_refs:
                    break

                self._log(
                    "TOC exploration round",
                    {"plan_index": plan_index, "round": attempt, "toc_refs": fresh_refs},
                    kind="state",
                )
                raw_evidence = self._extract_toc_entries(catalog, fresh_refs)
                prepared_evidence = self._prepare_evidence_items_for_review(
                    self._request_text_with_runtime_date(user_text),
                    effective_plan,
                    corp,
                    report,
                    raw_evidence,
                )
                accumulated_evidence.extend(prepared_evidence)
                review = self._review_evidence(
                    self._request_text_with_runtime_date(user_text),
                    effective_plan,
                    corp,
                    report,
                    catalog,
                    [item["toc_ref"] for item in accumulated_evidence],
                    accumulated_evidence,
                )
                if review["status"] == "OKAY":
                    sufficiency_ok = True
                    break
                if attempt == 3:
                    break
                selected_refs = review.get("toc_refs", [])

            materials = self._compose_plan_materials(
                user_text=user_text,
                corp=corp,
                plan=effective_plan,
                report=report,
                evidence_items=accumulated_evidence,
                sufficiency_ok=sufficiency_ok,
            )
            failure_type, failure_reason = self._classify_plan_outcome(
                request_mode=request_mode,
                materials=materials,
                evidence_items=accumulated_evidence,
                sufficiency_ok=sufficiency_ok,
            )
            kept = {
                "plan_index": plan_index,
                "status": "ok" if materials else "failed",
                "corp": corp,
                "plan": effective_plan,
                "report": report,
                "evidence_items": accumulated_evidence,
                "materials": materials,
                "attempts_used": attempts_used,
                "sufficiency_ok": sufficiency_ok,
                "failure_type": failure_type or "",
                "failure_reason": failure_reason or "",
            }
            self._log("Plan result kept", kept, kind="state")
            return kept
        except Exception as e:
            failure_reason = str(e)
            failure_type = self._classify_exception_failure(failure_reason)
            kept = {
                "plan_index": plan_index,
                "status": "failed",
                "corp": empty_corp,
                "plan": plan,
                "report": empty_report,
                "evidence_items": [],
                "materials": [],
                "attempts_used": 0,
                "sufficiency_ok": False,
                "failure_type": failure_type,
                "failure_reason": failure_reason,
            }
            self._log("Plan failed", kept, kind="error")
            return kept

    def find_dart_material(self, query: str) -> Dict[str, Any]:
        """
        외부 function calling / tools 연동을 위한 public one-shot DART retrieval tool입니다.
        Public one-shot DART retrieval tool for external function-calling integrations.

        Returns:
            {
                "ok": bool,
                "text": str,
                "source_paths": List[str],
                "error": str,  # only when ok is False
            }
        """
        query_text = str(query or "").strip()
        if not query_text:
            raise ValueError("query is required")

        collected = self._collect_wrapped_results(query_text)
        text = str(collected.get("text") or "").strip()
        source_paths = collected.get("source_paths") or []
        if not text:
            failure_reasons = []
            for item in collected.get("kept_results", []):
                summary = self._format_failure_summary(
                    str(item.get("failure_type") or ""),
                    str(item.get("failure_reason") or ""),
                ).strip()
                if summary and summary not in failure_reasons:
                    failure_reasons.append(summary)
            error = (
                "; ".join(failure_reasons[:3])
                if failure_reasons
                else "DART 공시에서 요청에 맞는 근거 자료를 찾지 못했습니다."
            )
            return {
                "ok": False,
                "text": "",
                "source_paths": [],
                "error": error,
            }

        result = {
            "ok": True,
            "text": text,
            "source_paths": source_paths,
        }
        self._log("Public tool result", result, kind="final")
        return result

    def ask(self, user_text: str) -> str:
        result = self.find_dart_material(user_text)
        if result.get("ok"):
            return str(result.get("text") or "").strip()
        return str(result.get("error") or "").strip()

    @staticmethod
    def _result_ok(result: Dict[str, Any]) -> bool:
        return bool(isinstance(result, dict) and result.get("ok"))

    def execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self.tool_registry:
            return {"ok": False, "error": f"Unknown tool: {name}"}

        try:
            return self.tool_registry[name](**args)
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def execute_public_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name != self.PUBLIC_TOOL_NAME:
            return {"ok": False, "text": "", "source_paths": [], "error": f"Unknown public tool: {name}"}

        try:
            return self.find_dart_material(**args)
        except Exception as e:
            return {"ok": False, "text": "", "source_paths": [], "error": str(e)}


class OpenAIDartAgent(BaseDartAgent):
    provider_name = "OpenAI"
    """
    OpenAI Responses API 루프 + custom function tools 실행기.
    """

    def __init__(
        self,
        openai_api_key: str,
        dart_api_key: str,
        model: str = OPENAI_DEFAULT_MODEL,
    ):
        if OpenAI is None:
            raise ImportError("openai package is required for provider='openai'")
        super().__init__(
            dart_api_key=dart_api_key,
            model=model,
        )
        self.client = OpenAI(api_key=openai_api_key)
        self.instructions = AGENT_INSTRUCTIONS_V2

    def execute_tool_call(self, tool_call: Any) -> Dict[str, Any]:
        result = self.execute_tool(tool_call.name, json.loads(tool_call.arguments))
        return {
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": json.dumps(result, ensure_ascii=False),
        }

    def _call_model_text(
        self,
        prompt: str,
        *,
        system_instruction: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=system_instruction or self.instructions,
            input=prompt,
        )
        return response.output_text


class GeminiDartAgent(BaseDartAgent):
    provider_name = "Gemini"
    """
    Gemini generateContent API + function calling 실행기.
    """

    GEMINI_API = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        gemini_api_key: str,
        dart_api_key: str,
        model: str = GEMINI_DEFAULT_MODEL,
        thinking_level: str = GEMINI_DEFAULT_FAST_THINKING_LEVEL,
        timeout: int = 60,
        fallback_api_keys: Optional[List[str]] = None,
    ):
        super().__init__(
            dart_api_key=dart_api_key,
            model=model,
        )
        self.api_key = gemini_api_key
        self.api_keys = [gemini_api_key] + [key for key in (fallback_api_keys or []) if key and key != gemini_api_key]
        self.thinking_level = thinking_level
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _generate_content(
        self,
        contents: List[Dict[str, Any]],
        model: str,
        thinking_level: Optional[str] = None,
        enable_tools: bool = True,
        tools_payload: Optional[List[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
        response_mime_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        last_error = None
        payload = {
            "systemInstruction": {"parts": [{"text": system_instruction or self.instructions}]},
            "contents": contents,
        }
        if enable_tools:
            active_tools = tools_payload or INTERNAL_GEMINI_TOOLS
            payload["tools"] = active_tools
            declared_names: List[str] = []
            for tool in active_tools:
                for declaration in tool.get("functionDeclarations", []):
                    name = declaration.get("name")
                    if name:
                        declared_names.append(name)
            if declared_names:
                payload["toolConfig"] = {
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": declared_names,
                    }
                }
        generation_config: Dict[str, Any] = {}
        if thinking_level:
            normalized_thinking_level = str(thinking_level).strip().lower()
            generation_config["thinkingConfig"] = {
                "thinkingLevel": normalized_thinking_level,
            }
        if response_mime_type:
            generation_config["responseMimeType"] = response_mime_type
        if generation_config:
            payload["generationConfig"] = generation_config
        for api_key in self.api_keys:
            response = self.session.post(
                f"{self.GEMINI_API}/{model}:generateContent",
                params={"key": api_key},
                json=payload,
                timeout=self.timeout,
            )
            if response.ok:
                self.api_key = api_key
                return response.json()

            last_error = requests.HTTPError(
                f"{response.status_code} Error: {response.text}",
                response=response,
            )
            if response.status_code not in {403, 429, 500, 503, 504}:
                raise last_error

        if last_error:
            raise last_error
        raise RuntimeError("Gemini request failed without a response")

    @staticmethod
    def _candidate_content(payload: Dict[str, Any]) -> Dict[str, Any]:
        candidates = payload.get("candidates", [])
        if not candidates:
            prompt_feedback = payload.get("promptFeedback")
            raise RuntimeError(f"Gemini returned no candidates: {prompt_feedback}")
        return candidates[0].get("content", {})

    @staticmethod
    def _content_text(content: Dict[str, Any]) -> str:
        texts = []
        for part in content.get("parts", []):
            if "text" in part and part["text"]:
                texts.append(part["text"])
        return "\n".join(texts).strip()

    @staticmethod
    def _function_calls(content: Dict[str, Any]) -> List[Dict[str, Any]]:
        calls = []
        for part in content.get("parts", []):
            function_call = part.get("functionCall")
            if function_call:
                calls.append(function_call)
        return calls

    def _call_model_text(
        self,
        prompt: str,
        *,
        system_instruction: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        payload = self._generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            model=self.model,
            thinking_level=self.thinking_level,
            enable_tools=False,
            system_instruction=system_instruction or self.instructions,
            response_mime_type="application/json" if json_mode else None,
        )
        content = self._candidate_content(payload)
        return self._content_text(content)


def _load_key_env(path: str = "key.env") -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not os.path.exists(path):
        return values

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _resolve_gemini_api_keys(key_env: Dict[str, str]) -> List[str]:
    ordered_candidates = [
        os.environ.get("GEMINI_API_KEY_PAY"),
        key_env.get("GEMINI_API_KEY_PAY"),
        os.environ.get("GEMINI_API_KEY"),
        key_env.get("GEMINI_API_KEY"),
        os.environ.get("GEMINI_API_KEY_FREE"),
        key_env.get("GEMINI_API_KEY_FREE"),
    ]
    keys = []
    for value in ordered_candidates:
        if value and value not in keys:
            keys.append(value)
    return keys


class DartToolRunner:
    """
    외부 LLM function calling 용 단일 DART 도구 실행기.
    TOOLS를 LLM API에 추가하고, function call이 오면 execute_tool로 실행한다.
    """

    def __init__(self, agent: BaseDartAgent):
        self.agent = agent
        self.tools = TOOLS

    def execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        return self.agent.execute_public_tool(name, args)


def dart_llm_tools_openai() -> List[Dict[str, Any]]:
    """OpenAI Responses API 등에 붙일 public DART tool schema 목록을 반환합니다."""
    return TOOLS


def dart_llm_tools_gemini() -> List[Dict[str, Any]]:
    """Gemini generateContent tools에 붙일 public DART tool declaration 목록을 반환합니다."""
    return PUBLIC_GEMINI_TOOLS


def dart_llm_tools(provider: str = "openai") -> List[Dict[str, Any]]:
    provider_normalized = str(provider or "openai").strip().lower()
    if provider_normalized == "gemini":
        return dart_llm_tools_gemini()
    return dart_llm_tools_openai()


def get_openai_public_tools() -> List[Dict[str, Any]]:
    return dart_llm_tools_openai()


def get_gemini_public_tools() -> List[Dict[str, Any]]:
    return dart_llm_tools_gemini()


def get_public_tools(provider: str = "openai") -> List[Dict[str, Any]]:
    return dart_llm_tools(provider)


def get_recent_filings_by_stock_code(
    stock_code: str,
    *,
    dart_api_key: Optional[str] = None,
    bgn_de: Optional[str] = None,
    end_de: Optional[str] = None,
    pblntf_ty: Optional[str] = None,
    pblntf_detail_ty: Optional[str] = None,
    last_reprt_at: str = "Y",
    limit: int = 20,
) -> Dict[str, Any]:
    """
    종목코드 기준 최근 공시 목록 helper / Recent filings helper by stock code.
    """
    resolved_dart_key = dart_api_key or os.environ.get("DART_API_KEY") or _load_key_env().get("opendart_key")
    if not resolved_dart_key:
        raise KeyError("DART_API_KEY not found. Set env var or pass dart_api_key.")
    service = DartReportToolService(dart_api_key=resolved_dart_key)
    return service.search_recent_filings_by_stock_code(
        stock_code=stock_code,
        bgn_de=bgn_de,
        end_de=end_de,
        pblntf_ty=pblntf_ty,
        pblntf_detail_ty=pblntf_detail_ty,
        last_reprt_at=last_reprt_at,
        limit=limit,
    )


def create_dart_agent(
    provider: str,
    dart_api_key: Optional[str] = None,
    key_env: Optional[Dict[str, str]] = None,
    model: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    fallback_api_keys: Optional[List[str]] = None,
):
    """
    내부 DART 에이전트를 생성합니다. 외부에는 model 하나만 노출합니다.
    Create the internal DART agent. The public API exposes a single model argument.
    """
    provider_normalized = str(provider or "").strip().lower()
    resolved_key_env = key_env or _load_key_env()
    resolved_dart_key = dart_api_key or os.environ.get("DART_API_KEY") or resolved_key_env.get("opendart_key")
    if not resolved_dart_key:
        raise KeyError("DART_API_KEY not found. Set env var or pass dart_api_key.")

    if provider_normalized == "openai":
        openai_key = openai_api_key or os.environ.get("OPENAI_API_KEY") or resolved_key_env.get("openai_key")
        if not openai_key:
            raise KeyError("OPENAI_API_KEY not found. Set env var or pass openai_api_key.")
        chosen_model = model or OPENAI_DEFAULT_MODEL
        return OpenAIDartAgent(
            openai_api_key=openai_key,
            dart_api_key=resolved_dart_key,
            model=chosen_model,
        )

    if provider_normalized == "gemini":
        gemini_keys: List[str] = []
        for value in [gemini_api_key] + list(fallback_api_keys or []):
            if value and value not in gemini_keys:
                gemini_keys.append(value)
        for value in _resolve_gemini_api_keys(resolved_key_env):
            if value and value not in gemini_keys:
                gemini_keys.append(value)
        if not gemini_keys:
            raise KeyError(
                "Gemini API key not found. Set GEMINI_API_KEY or pass gemini_api_key."
            )
        chosen_model = model or GEMINI_DEFAULT_MODEL
        return GeminiDartAgent(
            gemini_api_key=gemini_keys[0],
            dart_api_key=resolved_dart_key,
            model=chosen_model,
            fallback_api_keys=gemini_keys[1:],
        )

    raise ValueError("provider must be 'openai' or 'gemini'")


def create_dart_tool_runner(
    provider: str = "openai",
    *,
    key_env: Optional[Dict[str, str]] = None,
    model: Optional[str] = None,
    dart_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    fallback_api_keys: Optional[List[str]] = None,
) -> DartToolRunner:
    """
    외부 function calling/tools 연동용 DART runner를 생성합니다.
    Create a DART runner for external function-calling or tool integrations.
    """
    agent = create_dart_agent(
        provider=provider,
        dart_api_key=dart_api_key,
        key_env=key_env,
        model=model,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        fallback_api_keys=fallback_api_keys,
    )
    return DartToolRunner(agent)


def run_dart_tool_call(
    name: str,
    args: Dict[str, Any],
    *,
    provider: str = "openai",
    key_env: Optional[Dict[str, str]] = None,
    model: Optional[str] = None,
    dart_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    fallback_api_keys: Optional[List[str]] = None,
) -> DartMaterialResult:
    """
    고급 모드 실행 함수입니다. raw tool result를 직접 반환합니다.
    Advanced-mode execution function that returns the raw tool result directly.
    """
    runner = create_dart_tool_runner(
        provider=provider,
        key_env=key_env,
        model=model,
        dart_api_key=dart_api_key,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        fallback_api_keys=fallback_api_keys,
    )
    return runner.execute_tool(name, args)


def execute_dart_tool_call(
    name: str,
    args: Dict[str, Any],
    *,
    provider: str = "openai",
    key_env: Optional[Dict[str, str]] = None,
    model: Optional[str] = None,
    dart_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    fallback_api_keys: Optional[List[str]] = None,
) -> DartMaterialResult:
    return run_dart_tool_call(
        name,
        args,
        provider=provider,
        key_env=key_env,
        model=model,
        dart_api_key=dart_api_key,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        fallback_api_keys=fallback_api_keys,
    )


def dart_tool_gemini(
    *,
    dart_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    model: Optional[str] = None,
    fallback_api_keys: Optional[List[str]] = None,
    key_env: Optional[Dict[str, str]] = None,
    on_result: Optional[Callable[[DartMaterialResult], None]] = None,
) -> Callable[[str], DartMaterialResult]:
    """
    Gemini SDK의 tools=[...]에 바로 넣을 수 있는 callable DART tool을 만듭니다.
    Create a Gemini SDK-compatible callable DART tool for tools=[...].

    필요하면 on_result 콜백으로 raw tool result를 회수할 수 있습니다.
    If needed, you can collect the raw tool result through the on_result callback.
    """

    def find_dart_material(query: str) -> DartMaterialResult:
        result = run_dart_tool_call(
            "find_dart_material",
            {"query": query},
            provider="gemini",
            model=model,
            dart_api_key=dart_api_key,
            gemini_api_key=gemini_api_key,
            fallback_api_keys=fallback_api_keys,
            key_env=key_env,
        )
        if callable(on_result):
            on_result(result)
        return result

    return find_dart_material


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini"],
        default=os.environ.get("LLM_PROVIDER", "openai"),
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--question",
        default="삼성전자 2024년 사업보고서에서 연결재무제표 주석의 현금흐름표 핵심 수치를 짧게 요약해줘.",
    )
    args = parser.parse_args()

    key_env = _load_key_env()
    dart_key = os.environ.get("DART_API_KEY") or key_env.get("opendart_key")
    if not dart_key:
        raise KeyError("DART_API_KEY not found. Set env var or opendart_key in key.env.")

    agent = create_dart_agent(
        provider=args.provider,
        dart_api_key=dart_key,
        key_env=key_env,
        model=args.model,
    )
    answer = agent.ask(args.question)
    print(answer)
