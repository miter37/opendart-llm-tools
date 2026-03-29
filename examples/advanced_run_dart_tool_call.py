import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from opendart_llm_tools import run_dart_tool_call

result = run_dart_tool_call(
    "find_dart_material",
    {
        "query": "삼성전자 2024 사업보고서에서 유형자산 관련 감가상각비 설명 찾아줘",
    },
    provider="gemini",
    dart_api_key=os.environ["DART_API_KEY"],
    gemini_api_key=os.environ["GEMINI_API_KEY"],
)

print(result["ok"])
print(result["text"])
print(result["source_paths"])
