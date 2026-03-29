# opendart-llm-tools

OpenAI, Gemini 같은 LLM API 워크플로우에 간단한 함수 추가만으로 한국 DART 공시 검색을 붙일 수 있는 Python 도구입니다.  
opendart-llm-tools is a Python toolkit that lets apps using OpenAI or Gemini APIs add Korean DART filing retrieval with a simple function integration.

별도의 서버를 따로 띄우지 않고, 함수 import와 tool 연결만으로 DART 조회를 사용할 수 있습니다.  
No separate server is required; you can enable DART retrieval by importing functions and attaching a tool to your existing LLM API client.

근거 텍스트와 `source_paths`를 함께 반환하므로, 어떤 공시와 어느 섹션을 근거로 썼는지 추적할 수 있습니다.  
It returns raw evidence text together with `source_paths`, so you can see which filings and sections were used.

예제 스크립트와 데모 웹앱도 함께 포함되어 있습니다.  
Example scripts and demo web apps are included.

## Quick Start

아래 3줄 정도만 추가하면 Gemini API 클라이언트에 DART tool을 바로 붙일 수 있습니다.

```python
from opendart_llm_tools import dart_tool_gemini
tool = dart_tool_gemini( dart_api_key=os.environ["DART_API_KEY"],    gemini_api_key=os.environ["GEMINI_API_KEY"], )
#config = types.GenerateContentConfig(tools=[tool])   # gemini api config 내에  tools 추가
```

## Install

```bash
pip install -e .
```

저장소를 직접 내려받아 개발 모드로 설치할 때 사용하는 방식입니다.  
Use this when you clone the repository and want an editable local install.

## API Keys

필수:

```bash
# Required
export DART_API_KEY="your_dart_key"
```

OpenAI 예제:

```bash
export OPENAI_API_KEY="your_openai_key"
```

Gemini 예제:

```bash
export GEMINI_API_KEY="your_gemini_key"
```

공개 저장소에는 `key.env`를 올리지 말고, 로컬 개발용으로만 사용하세요.  
Do not commit `key.env` to a public repository; keep it only for local development.

## Minimal Examples

### Gemini

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

### OpenAI

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

## SDK Tool Calling

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
    model="gemini-3.1-flash-lite-preview",
    contents="삼성전자 최근 사업보고서에서 유형자산 관련 감가상각비 설명 찾아줘",
    config=types.GenerateContentConfig(tools=[tool]),
)

print(response.text)
```

### OpenAI Responses API

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
    arguments = json.loads(item.arguments or "{}")
    tool_result = run_dart_tool_call(
        item.name,
        arguments,
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

## What It Returns

```python
{
    "ok": True,
    "text": "...",
    "source_paths": ["회사명_연도_문서/섹션"],
    "error": ""
}
```

- `text`: tool이 찾은 근거 텍스트
- `source_paths`: 어떤 공시와 어느 섹션을 썼는지 보여주는 경로 목록
- `error`: 실패 시 오류 메시지

## Project Layout

```text
src/opendart_llm_tools/    # package source
examples/                  # minimal runnable examples
demo/                      # demo web apps
use_cases/                 # saved example questions and outputs
```

## Demo Apps

- `demo/web_dart_test_app.py`
- `demo/web_financial_app.py`

브라우저에서 직접 질문을 넣어보고 로그 흐름까지 확인할 수 있는 데모 앱입니다.  
These demo apps let you try the workflow in a browser and inspect the retrieval flow.

## Use Cases

`use_cases/` 폴더에는 실제 데모 실행에서 얻은 질문/응답 예시 텍스트가 들어 있습니다. 사용성이나 결과 형태를 빠르게 확인하고 싶다면 이 폴더를 참고하세요.  
The `use_cases/` folder contains saved question/answer examples from real demo runs. Check it if you want quick examples of actual usage and output format.
