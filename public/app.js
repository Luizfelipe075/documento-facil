const form = document.querySelector("#question-form");
const questionInput = document.querySelector("#question");
const charCount = document.querySelector("#char-count");
const submitButton = document.querySelector("#submit-button");
const activity = document.querySelector("#activity");
const activityTitle = document.querySelector("#activity-title");
const result = document.querySelector("#result");
const answerText = document.querySelector("#answer-text");
const sourcesSection = document.querySelector("#sources-section");
const sourcesList = document.querySelector("#sources-list");
const sourceCount = document.querySelector("#source-count");
const groundedBadge = document.querySelector("#grounded-badge");
const errorBox = document.querySelector("#error-box");
const terminalLog = document.querySelector("#terminal-log");

function addLog(message, level = "info") {
  const line = document.createElement("div");
  line.className = `log-line${level === "error" ? " log-error" : ""}`;
  const time = document.createElement("span");
  time.className = "log-time";
  time.textContent = new Intl.DateTimeFormat("pt-BR", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date());
  line.append(time, document.createTextNode(message));
  terminalLog.append(line);
  terminalLog.scrollTop = terminalLog.scrollHeight;
}

function showBackendEvents(events = []) {
  const labels = {
    request_received: "Requisição recebida pela função Netlify.",
    web_search_started: "Busca web do Gemini iniciada.",
    gemini_response_received: "Resposta recebida do Gemini.",
    insufficient_grounding: "A busca não retornou fontes verificáveis suficientes.",
    answer_ready: "Resposta pronta com fontes verificáveis.",
    configuration_error: "A chave Gemini não está configurada no servidor.",
    generation_error: "Erro ao gerar a resposta no Gemini.",
  };
  for (const event of events) {
    if (labels[event.stage]) addLog(labels[event.stage], event.stage.endsWith("error") ? "error" : "info");
  }
}

function setStep(name, state) {
  const item = document.querySelector(`[data-step="${name}"]`);
  if (!item) return;
  item.classList.toggle("done", state === "done");
  item.classList.toggle("current", state === "current");
  item.querySelector(".step-state").textContent = state === "done" ? "concluído" : state === "current" ? "em andamento" : "";
  if (state === "done") item.querySelector(".step-icon").textContent = "✓";
  else item.querySelector(".step-icon").textContent = String([...item.parentElement.children].indexOf(item) + 1);
}

function showSources(sources = []) {
  sourcesList.replaceChildren();
  const safeSources = sources.filter((source) => {
    try {
      const url = new URL(source.url);
      return url.protocol === "https:" || url.protocol === "http:";
    } catch {
      return false;
    }
  });

  for (const source of safeSources) {
    const url = new URL(source.url);
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.className = "source-card";
    link.href = url.href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    const glyph = document.createElement("span");
    glyph.className = "source-glyph";
    glyph.setAttribute("aria-hidden", "true");
    glyph.textContent = "↗";
    const copy = document.createElement("span");
    copy.className = "source-copy";
    const title = document.createElement("span");
    title.className = "source-title";
    title.textContent = source.title || url.hostname;
    const host = document.createElement("span");
    host.className = "source-host";
    host.textContent = url.hostname;
    copy.append(title, host);
    link.append(glyph, copy);
    item.append(link);
    sourcesList.append(item);
  }

  sourcesSection.hidden = safeSources.length === 0;
  sourceCount.textContent = safeSources.length ? `${safeSources.length} ${safeSources.length === 1 ? "fonte" : "fontes"}` : "";
  groundedBadge.hidden = safeSources.length === 0;
}

questionInput.addEventListener("input", () => {
  charCount.textContent = `${questionInput.value.length} / 1200`;
});

for (const chip of document.querySelectorAll(".example-chip")) {
  chip.addEventListener("click", () => {
    questionInput.value = chip.dataset.question || "";
    questionInput.dispatchEvent(new Event("input"));
    questionInput.focus();
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  submitButton.disabled = true;
  activity.hidden = false;
  activityTitle.textContent = "Pesquisando na web e preparando a resposta…";
  result.hidden = true;
  errorBox.hidden = true;
  setStep("search", "current");
  setStep("ground", "idle");
  setStep("answer", "idle");
  terminalLog.replaceChildren();
  addLog("Pergunta enviada à função Netlify.");
  addLog("ETAPA 1: busca web com Grounding do Google ativa.");

  try {
    const response = await fetch("/api/answer", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const payload = await response.json().catch(() => ({
      error: response.status === 429
        ? "Muitas perguntas em pouco tempo. Aguarde um minuto e tente novamente."
        : `O servidor respondeu com erro HTTP ${response.status}. Tente novamente.`,
    }));

    if (!response.ok) {
      showBackendEvents(payload.events);
      const error = new Error(payload.error || "A pesquisa não pôde ser concluída.");
      error.code = payload.code;
      throw error;
    }

    setStep("search", "done");
    setStep("ground", "done");
    setStep("answer", "done");
    showBackendEvents(payload.events);

    answerText.textContent = payload.answer || "Não encontrei uma resposta suficiente.";
    showSources(payload.sources || []);
    result.hidden = false;
    activityTitle.textContent = payload.grounded ? "Resposta pronta com fontes da web." : "Não foi possível confirmar uma resposta nas fontes.";
    document.querySelector(".terminal-panel").open = true;
    result.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    addLog(error.message || "Falha na solicitação.", "error");
    errorBox.textContent = error.message || "Erro de rede. Tente novamente.";
    errorBox.hidden = false;
    activityTitle.textContent = "A pesquisa falhou.";
    setStep("search", "idle");
    setStep("ground", "idle");
    setStep("answer", "idle");
    document.querySelector(".terminal-panel").open = true;
  } finally {
    submitButton.disabled = false;
  }
});
