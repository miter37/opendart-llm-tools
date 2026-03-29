const providerSelect = document.getElementById("providerSelect");
const questionInput = document.getElementById("questionInput");
const runBtn = document.getElementById("runBtn");
const statusPill = document.getElementById("statusPill");
const logBox = document.getElementById("logBox");
const resultBox = document.getElementById("resultBox");
const resultMeta = document.getElementById("resultMeta");
const appTitle = document.getElementById("appTitle");
const chatModelValue = document.getElementById("chatModelValue");
const toolModelValue = document.getElementById("toolModelValue");

let appConfig = {
  title: "dart_llm_tools",
  providers: {},
};

function formatResult(data) {
  const result = data?.result || {};
  const answer = data?.answer || "";
  const text = result.text || "";
  const sourcePaths = result.source_paths || data?.source_paths || [];
  const lines = [];

  lines.push("[answer]");
  lines.push(answer || "(empty)");
  lines.push("");
  lines.push("[text]");
  lines.push(text || "(empty)");
  lines.push("");
  lines.push("[source_paths]");
  if (sourcePaths.length) {
    for (const path of sourcePaths) {
      lines.push(`- ${path}`);
    }
  } else {
    lines.push("(none)");
  }

  if (result.error || data?.error) {
    lines.push("");
    lines.push("[error]");
    lines.push(result.error || data.error);
  }

  lines.push("");
  lines.push("[answer]: LLM API가 사용자에게 최종적으로 한 답변");
  lines.push("[text]: tool 함수가 반환한 근거로 사용할 텍스트 전체");
  lines.push("[source_paths]: tool 함수가 근거자료 작성에 참고한 문서와 항목위치");

  return lines.join("\n");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function setStatus(mode, text) {
  statusPill.className = `status-pill ${mode}`;
  statusPill.textContent = text;
}

function clearOutput() {
  logBox.innerHTML = "";
  resultBox.textContent = "";
}

function appendLog(item) {
  const entry = document.createElement("div");
  entry.className = `log-entry ${item.kind || "info"}`;

  const title = document.createElement("h3");
  title.className = "log-title";
  title.textContent = item.title || "Log";

  const detail = document.createElement("pre");
  detail.className = "log-detail";
  detail.textContent = item.detail || "";

  entry.appendChild(title);
  entry.appendChild(detail);
  logBox.appendChild(entry);
  logBox.scrollTop = logBox.scrollHeight;
}

function renderEmptyLogState() {
  logBox.innerHTML = `
    <div class="log-entry empty">
      <h3 class="log-title">No logs</h3>
      <pre class="log-detail">Run을 누르면 tool 호출 여부와 실행 과정이 여기에 표시됩니다.</pre>
    </div>
  `;
}

async function streamJsonLines(url, body, onEvent) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = "Request failed";
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch {}
    throw new Error(detail);
  }
  if (!response.body) {
    throw new Error("Streaming response body is unavailable");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      onEvent(JSON.parse(trimmed));
    }
  }

  const tail = buffer.trim();
  if (tail) {
    onEvent(JSON.parse(tail));
  }
}

function updateProviderMeta() {
  const provider = providerSelect.value || "gemini";
  const providerConfig = appConfig.providers?.[provider] || {};
  chatModelValue.textContent = providerConfig.chat_model || "-";
  toolModelValue.textContent = providerConfig.dart_tool_model || "-";
}

async function loadConfig() {
  const response = await fetch("/api/config");
  const data = await response.json();
  appConfig = data || appConfig;
  appTitle.textContent = appConfig.title || "dart_llm_tools";

  const providerEntries = Object.entries(appConfig.providers || {});
  providerSelect.innerHTML = providerEntries
    .map(([value, config]) => {
      const label = config.label || value;
      return `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`;
    })
    .join("");

  if (!providerEntries.length) {
    providerSelect.innerHTML = '<option value="gemini">Gemini</option>';
  }

  updateProviderMeta();
}

async function runLookup() {
  const provider = providerSelect.value;
  const question = questionInput.value.trim();

  if (!question) {
    setStatus("error", "Question required");
    resultMeta.textContent = "질문을 입력해 주세요.";
    return;
  }

  clearOutput();
  renderEmptyLogState();
  resultMeta.textContent = `${provider} request in progress`;
  setStatus("running", `Running ${provider}`);
  runBtn.disabled = true;

  try {
    let finalData = null;
    await streamJsonLines(
      "/api/run/stream",
      {
        provider,
        question,
      },
      (event) => {
        if (event.type === "log" && event.log) {
          if (logBox.querySelector(".log-entry.empty")) {
            logBox.innerHTML = "";
          }
          appendLog(event.log);
          return;
        }
        if (event.type === "result") {
          finalData = event;
        }
      }
    );

    const data = finalData || { ok: false, error: "No final result returned" };
    resultBox.textContent = formatResult(data);

    if (data.ok) {
      setStatus("done", `${provider} done`);
      const sourceCount = (data.result?.source_paths || data.source_paths || []).length;
      resultMeta.textContent = `provider=${data.provider} / chat_model=${data.chat_model} / dart_tool_model=${data.dart_tool_model} / tool_used=${data.tool_used ? "yes" : "no"} / sources=${sourceCount}`;
    } else {
      setStatus("error", `${provider} error`);
      resultMeta.textContent = data.error || "Execution failed";
      if (!logBox.children.length || logBox.querySelector(".log-entry.empty")) {
        logBox.innerHTML = "";
        appendLog({
          kind: "error",
          title: "Execution error",
          detail: data.error || "Execution failed",
        });
      }
    }
  } catch (error) {
    logBox.innerHTML = "";
    appendLog({
      kind: "error",
      title: "Request failed",
      detail: error.message,
    });
    resultBox.textContent = formatResult({ ok: false, error: error.message });
    resultMeta.textContent = "HTTP or server error";
    setStatus("error", `${provider} error`);
  } finally {
    runBtn.disabled = false;
  }
}

providerSelect.addEventListener("change", updateProviderMeta);
runBtn.addEventListener("click", runLookup);
questionInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    runLookup();
  }
});

renderEmptyLogState();
loadConfig().catch((error) => {
  setStatus("error", "Config error");
  resultMeta.textContent = "앱 설정을 불러오지 못했습니다.";
  appendLog({
    kind: "error",
    title: "Config load failed",
    detail: error.message,
  });
});
