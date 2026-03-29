# opendart-llm-tools

한국어: `opendart-llm-tools`는 OpenAI, Gemini 같은 LLM에 한국 DART 공시 조회 기능을 붙일 수 있게 해주는 Python 도구입니다.  
English: `opendart-llm-tools` is a Python toolkit for adding Korean DART filing retrieval to LLMs such as OpenAI and Gemini.

한국어: 단 3줄 추가로, LLM API가 한국 DART 공시를 참고해 답하게 만들 수 있습니다.  
English: With just 3 extra lines, your LLM API client can answer using Korean DART filings.

```python
from opendart_llm_tools import dart_tool_gemini

tool = dart_tool_gemini(
    dart_api_key=os.environ["DART_API_KEY"],
    gemini_api_key=os.environ["GEMINI_API_KEY"],
)

# add to your API client
config=types.GenerateContentConfig(tools=[tool])
```

한국어: 이 패키지는 필요한 공시를 찾고, 관련 본문과 주석을 추출하고, 근거 텍스트와 출처를 구조화해서 반환합니다.  
English: The package finds relevant filings, extracts useful body or note text, and returns structured evidence text with source locations.

## What It Returns / 무엇을 반환하나요?

```python
{
    "ok": True,
    "text": "...",
    "source_paths": ["회사명_연도_보고서시기_문서/항목"],
    "error": ""
}
```

- 한국어: `text`는 tool 함수가 반환한 근거 텍스트 전체입니다.  
  English: `text` is the evidence text returned by the tool.
- 한국어: `source_paths`는 어떤 문서와 항목을 참고했는지 보여줍니다.  
  English: `source_paths` shows which documents and sections were used.

## Install / 설치

```bash
pip install -e .
```

한국어: 저장소에서 바로 개발하려면 위 명령을 사용하세요.  
English: Use the command above when developing directly from the repository.

## API Keys / API 키 설정

한국어: 이 패키지는 사용자의 DART/OpenAI/Gemini API key를 환경변수 또는 함수 인자로 받습니다.  
English: This package accepts the user's DART/OpenAI/Gemini API keys through environment variables or direct function arguments.

```bash
# Required
export DART_API_KEY="your_dart_key"

# For OpenAI examples
export OPENAI_API_KEY="your_openai_key"

# For Gemini examples
export GEMINI_API_KEY="your_gemini_key"
```

한국어: 공개 저장소에는 `key.env`를 올리지 말고, 로컬 개발용으로만 사용하세요.  
English: Do not commit `key.env` to a public repository; use it only for local development.

## Minimal Examples / 최소 예시

### Gemini Minimal

```python
import os
from opendart_llm_tools import run_dart_tool_call

result = run_dart_tool_call(
    "find_dart_material",
    {"query": "삼성전자 최근 사업보고서에서 유형자산 관련 감가상각비 설명 찾아줘"},
    provider="gemini",
    dart_api_key=os.environ["DART_API_KEY"],
    gemini_api_key=os.environ["GEMINI_API_KEY"],
)

print(result["ok"])
print(result["text"])
print(result["source_paths"])
```

### OpenAI Minimal

```python
import os
from opendart_llm_tools import run_dart_tool_call

result = run_dart_tool_call(
    "find_dart_material",
    {"query": "삼성전자 최근 사업보고서에서 유형자산 관련 감가상각비 설명 찾아줘"},
    provider="openai",
    dart_api_key=os.environ["DART_API_KEY"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
)

print(result["ok"])
print(result["text"])
print(result["source_paths"])
```

한국어: 위 예시는 가장 짧게 raw DART evidence를 직접 받아보는 방법입니다.  
English: The examples above are the shortest way to get raw DART evidence directly.

## SDK Tool Calling / SDK tool calling 예시

### Gemini

```python
import os
from google import genai
from google.genai import types

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
```

한국어: Gemini Python SDK에서는 `tools=[...]`에 callable tool을 바로 넣을 수 있습니다.  
English: In the Gemini Python SDK, you can pass a callable tool directly in `tools=[...]`.

### OpenAI

```python
import json
import os

from openai import OpenAI
from opendart_llm_tools import dart_llm_tools_openai, run_dart_tool_call

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
```

한국어: OpenAI 쪽은 tool schema를 붙이고, function call이 오면 `run_dart_tool_call(...)`로 실행하면 됩니다.  
English: With OpenAI, attach the tool schema and execute incoming function calls with `run_dart_tool_call(...)`.

## Advanced Mode / 고급 모드

```python
import os
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
```

한국어: 고급 모드에서는 raw evidence를 직접 다룰 수 있습니다.  
English: In advanced mode, you can handle raw evidence directly.

## How It Works / 동작 방식

1. 한국어: 바깥 LLM이 tool 사용 여부를 판단합니다.  
   English: The outer LLM decides whether to use the tool.
2. 한국어: 내부 worker LLM이 회사, 보고서, 섹션 후보를 계획합니다.  
   English: The internal worker LLM plans which company, filing, and sections to inspect.
3. 한국어: DART API와 viewer HTML을 통해 실제 공시 자료를 찾습니다.  
   English: The tool retrieves actual filings through the DART API and viewer HTML.
4. 한국어: 긴 본문은 chunk로 나눠 필요한 부분만 정리합니다.  
   English: Long excerpts are chunked so the relevant parts can be focused and kept.
5. 한국어: 충분한 근거가 모이면 `text`와 `source_paths`를 반환합니다.  
   English: Once enough evidence is gathered, the tool returns `text` and `source_paths`.

## Project Layout / 프로젝트 구조

- `src/opendart_llm_tools/`: 한국어: 실제 패키지 코드 / English: actual package code
- `examples/`: 한국어: 사용 예제 / English: usage examples
- `demo/`: 한국어: 데모 웹앱 / English: demo web app
- `pyproject.toml`: 한국어: 패키지 메타데이터 / English: package metadata

## Demo / 데모 웹앱

```bash
python -m uvicorn demo.web_dart_test_app:app --host 127.0.0.1 --port 8001
```

한국어: 브라우저에서 `http://127.0.0.1:8001`로 접속하세요.  
English: Open `http://127.0.0.1:8001` in your browser.
