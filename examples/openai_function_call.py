import json
import os
import sys
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dart_llm_tools import dart_llm_tools_openai, run_dart_tool_call

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

response = client.responses.create(
    model="gpt-5.4-mini",
    input="삼성전자 최근 사업보고서에서 유형자산 관련 감가상각비 설명 찾아줘",
    tools=dart_llm_tools_openai(),
)

for item in response.output:
    if item.type != "function_call":
        continue

    tool_result = run_dart_tool_call(
        item.name,
        json.loads(item.arguments),
        provider="openai",
        dart_api_key=os.environ["DART_API_KEY"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )

    response = client.responses.create(
        model="gpt-5.4-mini",
        previous_response_id=response.id,
        input=[
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(tool_result, ensure_ascii=False),
            }
        ],
    )

print(response.output_text)
