import os
import sys
from pathlib import Path

from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from opendart_llm_tools import dart_tool_gemini

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

tool = dart_tool_gemini(
    dart_api_key=os.environ["DART_API_KEY"],
    gemini_api_key=os.environ["GEMINI_API_KEY"],
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="삼성전자 최근 사업보고서에서 유형자산 관련 감가상각비 설명 찾아줘",
    config=types.GenerateContentConfig(
        tools=[tool],
    ),
)

print(response.text)
