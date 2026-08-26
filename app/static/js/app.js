(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const elements = {
    apiKey: $("#apiKey"), rememberApiKey: $("#rememberApiKey"), toggleKey: $("#toggleKey"),
    baseUrl: $("#baseUrl"), embeddingModel: $("#embeddingModel"), llmModel: $("#llmModel"),
    configBadge: $("#configBadge"), testConfigButton: $("#testConfigButton"),
    resetSettingsButton: $("#resetSettingsButton"), configTestResult: $("#configTestResult"),
    dropZone: $("#dropZone"), documentInput: $("#documentInput"), fileList: $("#fileList"),
    processButton: $("#processButton"), indexPulse: $("#indexPulse"),
    indexStatusText: $("#indexStatusText"), indexStats: $("#indexStats"),
    fileCount: $("#fileCount"), chunkCount: $("#chunkCount"), clearIndexButton: $("#clearIndexButton"),
    chatSubtitle: $("#chatSubtitle"), workflowTrack: $("#workflowTrack"),
    chatMessages: $("#chatMessages"), emptyState: $("#emptyState"), chatForm: $("#chatForm"),
    questionInput: $("#questionInput"), sendButton: $("#sendButton"), clearChatButton: $("#clearChatButton"),
    toast: $("#toast"), toastIcon: $("#toastIcon"), toastMessage: $("#toastMessage"),
  };

  const state = {
    files: [], messages: [], indexReady: false, serverKey: false, busy: false, toastTimer: null,
  };
  const maxFiles = Number(document.body.dataset.maxFiles || 10);
  const maxSizeMb = Number(document.body.dataset.maxFileSizeMb || 5);
  const storageKey = `${document.body.dataset.projectKey}:model-settings:v1`;
  const defaults = {
    baseUrl: elements.baseUrl.value,
    embeddingModel: elements.embeddingModel.value,
    llmModel: elements.llmModel.value,
  };

  function configured() {
    return Boolean(
      (elements.apiKey.value.trim() || state.serverKey) && elements.baseUrl.value.trim() &&
      elements.embeddingModel.value.trim() && elements.llmModel.value.trim(),
    );
  }

  function saveSettings() {
    const value = {
      baseUrl: elements.baseUrl.value.trim(), embeddingModel: elements.embeddingModel.value.trim(),
      llmModel: elements.llmModel.value.trim(), rememberApiKey: elements.rememberApiKey.checked,
    };
    if (value.rememberApiKey) value.apiKey = elements.apiKey.value.trim();
    try { localStorage.setItem(storageKey, JSON.stringify(value)); } catch (_) { /* unavailable */ }
  }

  function restoreSettings() {
    try {
      const value = JSON.parse(localStorage.getItem(storageKey) || "null");
      if (!value) return;
      if (value.baseUrl) elements.baseUrl.value = value.baseUrl;
      if (value.embeddingModel) elements.embeddingModel.value = value.embeddingModel;
      if (value.llmModel) elements.llmModel.value = value.llmModel;
      if (value.rememberApiKey && value.apiKey) {
        elements.rememberApiKey.checked = true;
        elements.apiKey.value = value.apiKey;
      }
    } catch (_) { localStorage.removeItem(storageKey); }
  }

  function resetSettings() {
    try { localStorage.removeItem(storageKey); } catch (_) { /* unavailable */ }
    elements.apiKey.value = "";
    elements.rememberApiKey.checked = false;
    elements.baseUrl.value = defaults.baseUrl;
    elements.embeddingModel.value = defaults.embeddingModel;
    elements.llmModel.value = defaults.llmModel;
    elements.configTestResult.hidden = true;
    updateControls();
    toast("已恢复默认设置。", "success");
  }

  function updateControls() {
    const ready = configured();
    elements.configBadge.className = `status-badge ${ready ? "ready" : "neutral"}`;
    elements.configBadge.querySelector("span").textContent = ready
      ? (state.serverKey && !elements.apiKey.value.trim() ? "服务端已配置" : "配置完成") : "待配置";
    elements.processButton.disabled = !ready || !state.files.length || state.busy;
    elements.testConfigButton.disabled = !ready || state.busy;
    elements.questionInput.disabled = !ready || !state.indexReady || state.busy;
    elements.sendButton.disabled = !ready || !state.indexReady || state.busy || !elements.questionInput.value.trim();
  }

  function headers(json = false) {
    const result = {};
    if (elements.apiKey.value.trim()) result["X-API-Key"] = elements.apiKey.value.trim();
    if (json) result["Content-Type"] = "application/json";
    return result;
  }

  async function parse(response) {
    let data = {};
    try { data = await response.json(); } catch (_) { /* empty body */ }
    if (!response.ok) {
      const detail = Array.isArray(data.detail) ? data.detail.map((item) => item.msg).join("；") : data.detail;
      throw new Error(detail || `请求失败（HTTP ${response.status}）`);
    }
    return data;
  }

  async function loadStatus() {
    try {
      const data = await parse(await fetch("/api/status"));
      state.serverKey = data.server_api_key_configured;
      state.indexReady = data.ready;
      elements.indexPulse.classList.toggle("ready", data.ready);
      elements.indexStatusText.textContent = data.ready ? "员工手册索引已就绪" : "等待上传员工手册";
      elements.indexStats.hidden = !data.ready;
      elements.clearIndexButton.hidden = !data.ready;
      if (data.ready) {
        elements.fileCount.textContent = data.file_count;
        elements.chunkCount.textContent = data.chunk_count;
        elements.chatSubtitle.textContent = `${data.file_count} 个手册 · ${data.chunk_count} 个章节 · ${data.embedding_model}`;
        elements.questionInput.placeholder = "询问休假、福利、绩效或入离职政策...";
      } else {
        elements.chatSubtitle.textContent = "完成设置并建立知识库后即可提问";
        elements.questionInput.placeholder = "请先构建员工手册索引...";
      }
      updateControls();
    } catch (error) { toast(error.message, "error"); }
  }

  function validateFiles(files) {
    const valid = [];
    const errors = [];
    for (const file of files) {
      const markdown = /\.(md|markdown)$/i.test(file.name);
      if (!markdown) errors.push(`${file.name} 不是 Markdown 文件`);
      else if (file.size > maxSizeMb * 1024 * 1024) errors.push(`${file.name} 超过 ${maxSizeMb} MB`);
      else valid.push(file);
    }
    if (valid.length > maxFiles) { valid.splice(maxFiles); errors.push(`单次最多 ${maxFiles} 个文件`); }
    if (errors.length) toast(errors.join("；"), "error");
    return valid;
  }

  function setFiles(files) {
    state.files = validateFiles(Array.from(files));
    elements.fileList.replaceChildren();
    state.files.forEach((file, index) => {
      const row = document.createElement("div"); row.className = "file-item";
      const badge = document.createElement("span"); badge.className = "file-badge"; badge.textContent = "MD";
      const copy = document.createElement("div"); copy.innerHTML = `<strong></strong><small></small>`;
      copy.querySelector("strong").textContent = file.name;
      copy.querySelector("small").textContent = `${(file.size / 1024).toFixed(1)} KB`;
      const remove = document.createElement("button"); remove.type = "button"; remove.textContent = "×";
      remove.addEventListener("click", () => { state.files.splice(index, 1); setFiles(state.files); });
      row.append(badge, copy, remove); elements.fileList.append(row);
    });
    updateControls();
  }

  async function buildIndex() {
    if (!configured() || !state.files.length) return;
    const body = new FormData();
    state.files.forEach((file) => body.append("files", file));
    body.append("embedding_model", elements.embeddingModel.value.trim());
    body.append("base_url", elements.baseUrl.value.trim());
    state.busy = true; elements.processButton.classList.add("loading");
    elements.processButton.querySelector("span").textContent = "正在切分并向量化..."; updateControls();
    try {
      const data = await parse(await fetch("/api/documents", { method: "POST", headers: headers(), body }));
      toast(`${data.file_count} 个文件已生成 ${data.chunk_count} 个章节向量。`, "success");
      state.files = []; elements.documentInput.value = ""; setFiles([]); clearConversation(); await loadStatus();
    } catch (error) { toast(error.message, "error"); }
    finally {
      state.busy = false; elements.processButton.classList.remove("loading");
      elements.processButton.querySelector("span").textContent = "构建 FAISS 索引"; updateControls();
    }
  }

  async function testConfig() {
    if (!configured()) return;
    state.busy = true; elements.configTestResult.hidden = true;
    elements.testConfigButton.querySelector("span").textContent = "正在连接..."; updateControls();
    try {
      const data = await parse(await fetch("/api/config/test", {
        method: "POST", headers: headers(true), body: JSON.stringify({
          base_url: elements.baseUrl.value.trim(), embedding_model: elements.embeddingModel.value.trim(),
          llm_model: elements.llmModel.value.trim(),
        }),
      }));
      elements.configTestResult.textContent = `连接成功 · ${data.embedding_dimensions} 维 · ${data.elapsed_ms} ms`;
      elements.configTestResult.className = "inline-result"; elements.configTestResult.hidden = false; toast(data.message, "success");
    } catch (error) {
      elements.configTestResult.textContent = error.message; elements.configTestResult.className = "inline-result error";
      elements.configTestResult.hidden = false; toast(error.message, "error");
    } finally {
      state.busy = false; elements.testConfigButton.querySelector("span").textContent = "测试当前配置"; updateControls();
    }
  }

  function addMessage(role, text, data = null) {
    if (elements.emptyState?.isConnected) elements.emptyState.remove();
    const article = document.createElement("article"); article.className = `message ${role}`;
    const avatar = document.createElement("div"); avatar.className = "avatar"; avatar.textContent = role === "user" ? "你" : "LC";
    const content = document.createElement("div"); content.className = "message-content";
    const bubble = document.createElement("div"); bubble.className = "bubble"; bubble.textContent = text; content.append(bubble);
    if (data) {
      const meta = document.createElement("div"); meta.className = "message-meta";
      meta.textContent = `${data.is_hr_related ? "HR 范围内" : "非 HR 问题"} · ${data.llm_model} · 检索 ${data.retrieved_chunks} 个章节`;
      content.append(meta);
      if (data.sources?.length) {
        const sources = document.createElement("div"); sources.className = "sources";
        data.sources.forEach((source) => {
          const tag = document.createElement("span"); tag.title = source.preview; tag.textContent = `${source.source} · ${source.heading}`; sources.append(tag);
        });
        content.append(sources);
      }
    }
    article.append(avatar, content); elements.chatMessages.append(article); elements.clearChatButton.hidden = false;
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight; return article;
  }

  function loadingMessage() {
    const message = addMessage("assistant", "");
    message.querySelector(".bubble").innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';
    return message;
  }

  function showWorkflow(nodes = []) {
    elements.workflowTrack.querySelectorAll(".workflow-node").forEach((node) => node.classList.remove("active", "visited"));
    const expanded = nodes.includes("classify") ? [...nodes.slice(0, 2), "branch", ...nodes.slice(2)] : nodes;
    expanded.forEach((name, index) => {
      const node = elements.workflowTrack.querySelector(`[data-node="${name}"]`);
      if (node) window.setTimeout(() => node.classList.add(index === expanded.length - 1 ? "active" : "visited"), index * 130);
    });
  }

  async function submit(event) {
    event.preventDefault();
    const question = elements.questionInput.value.trim();
    if (!question || !state.indexReady || state.busy) return;
    const outgoing = [...state.messages, { role: "user", content: question }].slice(-50);
    addMessage("user", question); elements.questionInput.value = ""; resize();
    state.busy = true; updateControls(); showWorkflow(["extract"]); const pending = loadingMessage();
    try {
      const data = await parse(await fetch("/api/chat", {
        method: "POST", headers: headers(true), body: JSON.stringify({ messages: outgoing,
          base_url: elements.baseUrl.value.trim(), embedding_model: elements.embeddingModel.value.trim(),
          llm_model: elements.llmModel.value.trim(), }),
      }));
      pending.remove(); addMessage("assistant", data.answer, data);
      state.messages = [...outgoing, { role: "assistant", content: data.answer }].slice(-50); showWorkflow(data.workflow);
    } catch (error) { pending.remove(); addMessage("assistant", `请求失败：${error.message}`); toast(error.message, "error"); }
    finally { state.busy = false; updateControls(); elements.questionInput.focus(); }
  }

  function clearConversation() {
    state.messages = []; showWorkflow([]); elements.chatMessages.replaceChildren();
    const empty = document.createElement("div"); empty.id = "emptyState"; empty.className = "empty-state compact";
    empty.innerHTML = `<div class="empty-orbit"><span>HR</span></div><span class="empty-kicker">${state.indexReady ? "INDEX READY" : "LANGCHAIN ADVANCED RAG"}</span><h3>${state.indexReady ? "知识库已就绪，可以开始提问" : "等待构建员工手册索引"}</h3><p>${state.indexReady ? "系统会先判断 HR 相关性，再选择检索回答或直接拒答。" : "完成左侧模型配置并上传 Markdown 员工手册。"}</p>`;
    elements.chatMessages.append(empty); elements.emptyState = empty; elements.clearChatButton.hidden = true;
  }

  async function clearIndex() {
    if (!window.confirm("确定清除当前 FAISS 索引吗？原始 Markdown 文件不会被删除。")) return;
    try { const data = await parse(await fetch("/api/index", { method: "DELETE" })); await loadStatus(); clearConversation(); toast(data.message, "success"); }
    catch (error) { toast(error.message, "error"); }
  }

  function resize() { elements.questionInput.style.height = "auto"; elements.questionInput.style.height = `${Math.min(elements.questionInput.scrollHeight, 130)}px`; updateControls(); }
  function toast(message, type) {
    clearTimeout(state.toastTimer); elements.toast.className = `toast visible ${type === "error" ? "error" : ""}`;
    elements.toastIcon.textContent = type === "error" ? "!" : "✓"; elements.toastMessage.textContent = message;
    state.toastTimer = setTimeout(() => elements.toast.classList.remove("visible"), 4200);
  }

  restoreSettings(); loadStatus(); updateControls();
  elements.toggleKey.addEventListener("click", () => {
    const reveal = elements.apiKey.type === "password"; elements.apiKey.type = reveal ? "text" : "password";
    elements.toggleKey.textContent = reveal ? "隐藏" : "显示";
  });
  [elements.apiKey, elements.baseUrl, elements.embeddingModel, elements.llmModel].forEach((input) => input.addEventListener("input", () => { saveSettings(); elements.configTestResult.hidden = true; updateControls(); }));
  elements.rememberApiKey.addEventListener("change", () => { saveSettings(); updateControls(); });
  elements.resetSettingsButton.addEventListener("click", resetSettings);
  elements.testConfigButton.addEventListener("click", testConfig);
  elements.documentInput.addEventListener("change", (event) => setFiles(event.target.files));
  ["dragenter", "dragover"].forEach((name) => elements.dropZone.addEventListener(name, (event) => { event.preventDefault(); elements.dropZone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((name) => elements.dropZone.addEventListener(name, (event) => { event.preventDefault(); elements.dropZone.classList.remove("dragging"); if (name === "drop") setFiles(event.dataTransfer.files); }));
  elements.processButton.addEventListener("click", buildIndex);
  elements.clearIndexButton.addEventListener("click", clearIndex);
  elements.chatForm.addEventListener("submit", submit);
  elements.clearChatButton.addEventListener("click", clearConversation);
  elements.questionInput.addEventListener("input", resize);
  elements.questionInput.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); elements.chatForm.requestSubmit(); } });
  document.querySelectorAll("[data-question]").forEach((button) => button.addEventListener("click", () => { elements.questionInput.value = button.dataset.question; resize(); elements.questionInput.focus(); }));
})();
