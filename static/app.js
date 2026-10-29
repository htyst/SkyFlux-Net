(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const order = ["5min", "1h", "2h", "1d", "3d"];
  const labels = {
    "5min": "5 分钟",
    "1h": "1 小时",
    "2h": "2 小时",
    "1d": "1 天",
    "3d": "3 天",
  };
  const state = { mode: "auto", response: null, selected: null, csv: "" };
  const fmt = (value) => {
    const n = Number(value);
    if (!Number.isFinite(n)) return "—";
    if (n === 0) return "0";
    return n >= 10000
      ? n.toExponential(2)
      : n >= 100
        ? n.toFixed(1)
        : n >= 1
          ? n.toFixed(2)
          : n.toPrecision(3);
  };
  const time = (value) => (value || "").replace("T", " ").slice(0, 19);
  const status = (message, kind = "") => {
    $("form-status").textContent = message;
    $("form-status").className = `form-status ${kind}`;
  };
  const alarmClass = (value) =>
    value === "红色"
      ? "red"
      : value === "橙色"
        ? "orange"
        : value === "黄色"
          ? "yellow"
          : "";
  function setMode(mode) {
    state.mode = mode;
    for (const item of ["auto", "direct"]) {
      const el = $(`mode-${item}`);
      el.classList.toggle("active", item === mode);
      el.setAttribute("aria-pressed", String(item === mode));
    }
    $("direct-options").classList.toggle("hidden", mode !== "direct");
    $("mode-hint").textContent =
      mode === "auto"
        ? "导入连续的五分钟观测，平台将按完整时间区间聚合并尝试生成五个尺度的预测。"
        : "直接导入所选尺度的连续观测。一次只预测该尺度；无需先转换成五分钟数据。";
  }
  function setCSV(value, name = "") {
    state.csv = value;
    if ($("csv-input").value !== value) $("csv-input").value = value;
    const lines = value
      ? Math.max(0, value.split(/\r?\n/).filter(Boolean).length - 1)
      : 0;
    $("data-meta").textContent = value
      ? `${name ? name + " · " : ""}约 ${lines.toLocaleString()} 行 · ${(new Blob([value]).size / 1024).toFixed(1)} KB`
      : "尚未选择数据";
    if (value) status("数据已准备，可以运行预测");
  }
  async function readFile(file) {
    if (!file) return;
    if (file.size > 23 * 1024 * 1024) {
      status("文件超过 23 MB，请缩小数据范围后再试", "error");
      return;
    }
    try {
      const value = await file.text();
      setCSV(value.replace(/^\uFEFF/, ""), file.name);
      $("upload-title").textContent = file.name;
      $("upload-subtitle").textContent =
        "已读取到浏览器内存，点击“运行预测”进行本地推理";
    } catch (err) {
      status(`读取文件失败：${err.message}`, "error");
    }
  }
  async function getJSON(url, options) {
    const response = await fetch(url, options);
    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error("本地服务返回了无效数据");
    }
    if (!response.ok)
      throw new Error(data.error || `请求失败：${response.status}`);
    return data;
  }
  function drawResults(response) {
    state.response = response;
    const found = order.filter(
      (key) => response.results && response.results[key],
    );
    state.selected = found[0] || null;
    $("results-empty").classList.toggle(
      "hidden",
      found.length > 0 || Object.keys(response.errors || {}).length > 0,
    );
    $("result-cards").classList.remove("hidden");
    $("result-cards").replaceChildren();
    $("export-button").classList.toggle("hidden", !found.length);
    const showKeys =
      response.mode === "direct"
        ? [found[0] || Object.keys(response.errors || {})[0]].filter(Boolean)
        : order;
    for (const key of showKeys) {
      const result = response.results && response.results[key];
      const card = document.createElement(result ? "button" : "div");
      card.className = `result-card${result ? "" : " error"}${state.selected === key ? " selected" : ""}`;
      if (result) {
        card.type = "button";
        card.addEventListener("click", () => selectResult(key));
      }
      const top = document.createElement("div");
      top.className = "card-top";
      const label = document.createElement("strong");
      label.textContent = labels[key] || key;
      const code = document.createElement("span");
      code.textContent = key.toUpperCase();
      top.append(label, code);
      card.append(top);
      if (result) {
        const next = result.predictions[0];
        const amount = document.createElement("div");
        const val = document.createElement("strong");
        val.className = "card-value";
        val.textContent = fmt(next.flux);
        const unit = document.createElement("span");
        unit.className = "card-unit";
        unit.textContent = "pfu";
        amount.append(val, unit);
        card.append(amount);
        const alarm = document.createElement("span");
        alarm.className = `alarm-pill ${alarmClass(next.alarm)}`;
        alarm.textContent = next.alarm;
        card.append(alarm);
        const bottom = document.createElement("div");
        bottom.className = "card-bottom";
        const baseline = document.createElement("span");
        baseline.append("持续性基线 ");
        const b = document.createElement("strong");
        b.textContent = `${fmt(result.baseline_flux)} pfu`;
        baseline.append(b);
        const timestamp = document.createElement("span");
        timestamp.textContent = time(next.datetime);
        bottom.append(baseline, timestamp);
        card.append(bottom);
      } else {
        const error = document.createElement("div");
        error.className = "card-error";
        error.textContent = response.errors?.[key] || "此尺度没有预测结果";
        card.append(error);
      }
      $("result-cards").append(card);
    }
    $("detail-panel").classList.toggle("hidden", !state.selected);
    if (state.selected) selectResult(state.selected);
    $("results").scrollIntoView({ behavior: "smooth", block: "start" });
  }
  function selectResult(key) {
    const result = state.response?.results?.[key];
    if (!result) return;
    state.selected = key;
    for (const card of $("result-cards").children)
      card.classList.toggle(
        "selected",
        card.querySelector(".card-top strong")?.textContent === labels[key],
      );
    $("detail-title").textContent = `${labels[key]} · 观测与下一时段预测`;
    $("detail-time").textContent =
      `最后观测 ${time(result.last_observation)} · 预测 ${time(result.predictions[0]?.datetime)} · ${result.timezone}`;
    chart(result);
  }
  function chart(result) {
    const host = $("chart");
    const history = (result.recent_series || []).filter(
      (x) => Number.isFinite(x.flux) && x.flux >= 0,
    );
    const prediction = result.predictions[0];
    if (!history.length || !prediction) {
      host.textContent = "没有可显示的数据";
      return;
    }
    const values = [...history.map((x) => x.flux), prediction.flux, 10].map(
      (x) => Math.log10(Math.max(x, 0.001)),
    );
    const low = Math.floor(Math.min(...values) - 0.35),
      high = Math.ceil(Math.max(...values) + 0.35);
    const W = 1000,
      H = 260,
      left = 55,
      right = 965,
      top = 16,
      bottom = 207;
    const x = (i) => left + (i * (right - left)) / history.length;
    const y = (v) =>
      bottom -
      ((Math.log10(Math.max(v, 0.001)) - low) / Math.max(1, high - low)) *
        (bottom - top);
    const path = history
      .map(
        (point, i) =>
          `${i ? "L" : "M"}${x(i).toFixed(2)},${y(point.flux).toFixed(2)}`,
      )
      .join(" ");
    const last = history[history.length - 1];
    const forecastX = x(history.length),
      forecastY = y(prediction.flux);
    const thresholdY = y(10);
    const ticks = [];
    for (let tick = low; tick <= high; tick++) {
      const ty =
        bottom - ((tick - low) / Math.max(1, high - low)) * (bottom - top);
      ticks.push(
        `<line x1="${left}" y1="${ty}" x2="${right}" y2="${ty}" stroke="#335662" stroke-width="1" opacity=".55"/><text x="${left - 10}" y="${ty + 4}" text-anchor="end" fill="#789ba5" font-size="10">10^${tick}</text>`,
      );
    }
    host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${labels[result.resolution]} P10 观测曲线和下一时段预测，对数纵轴"><defs><linearGradient id="linefill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#8cdef2" stop-opacity=".23"/><stop offset="100%" stop-color="#8cdef2" stop-opacity="0"/></linearGradient></defs>${ticks.join("")}<line x1="${left}" y1="${thresholdY}" x2="${right}" y2="${thresholdY}" stroke="#ffc879" stroke-width="1.5" stroke-dasharray="6 5"/><path d="${path} L${x(history.length - 1).toFixed(2)},${bottom} L${left},${bottom} Z" fill="url(#linefill)"/><path d="${path}" fill="none" stroke="#8cdef2" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/><line x1="${x(history.length - 1)}" y1="${y(last.flux)}" x2="${forecastX}" y2="${forecastY}" stroke="#69e5cf" stroke-width="2.6" stroke-dasharray="5 5"/><line x1="${forecastX}" y1="${top}" x2="${forecastX}" y2="${bottom}" stroke="#69e5cf" opacity=".4" stroke-dasharray="3 5"/><circle cx="${forecastX}" cy="${forecastY}" r="6" fill="#69e5cf" stroke="#0f2b36" stroke-width="3"/><text x="${left}" y="${H - 16}" fill="#789ba5" font-size="10">${time(history[0].datetime).slice(0, 16)}</text><text x="${right}" y="${H - 16}" text-anchor="end" fill="#a9ded6" font-size="10">${time(prediction.datetime).slice(0, 16)}  →</text><text x="${right - 4}" y="${Math.max(top + 12, thresholdY - 7)}" text-anchor="end" fill="#ffc879" font-size="10">黄色阈值 10 pfu</text></svg>`;
  }
  async function predict() {
    const csv = $("csv-input").value.trim();
    if (!csv) {
      status("请先上传或粘贴 CSV 数据", "error");
      $("csv-input").focus();
      return;
    }
    $("predict-button").disabled = true;
    status("正在验证数据并运行本地模型…");
    try {
      const data = await getJSON("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          csv,
          mode: state.mode,
          resolution: $("resolution").value,
          timezone: $("timezone").value,
          historical: $("historical").checked,
        }),
      });
      drawResults(data);
      const count = Object.keys(data.results || {}).length;
      status(
        count
          ? `完成 ${count} 个尺度预测${Object.keys(data.errors || {}).length ? "；其他尺度见下方说明" : ""}`
          : "没有可用预测，请查看各尺度的数据提示",
        count ? "success" : "error",
      );
    } catch (err) {
      status(err.message, "error");
    } finally {
      $("predict-button").disabled = false;
    }
  }
  async function demo() {
    $("demo-button").disabled = true;
    status("正在载入历史示例并运行模型…");
    try {
      const response = await getJSON("/api/demo");
      $("historical").checked = true;
      drawResults(response);
      const count = Object.keys(response.results || {}).length;
      status(
        `历史示例已完成 ${count} 个尺度预测。示例数据截至 2025 年 2 月。`,
        count ? "success" : "error",
      );
    } catch (err) {
      status(`示例载入失败：${err.message}`, "error");
    } finally {
      $("demo-button").disabled = false;
    }
  }
  function exportCSV() {
    const response = state.response;
    if (!response) return;
    const rows = [
      [
        "resolution",
        "last_observation",
        "prediction_datetime",
        "step",
        "model_pfu",
        "alarm",
        "persistence_pfu",
        "timezone",
        "historical",
      ],
    ];
    for (const key of order) {
      const result = response.results?.[key];
      if (!result) continue;
      for (const p of result.predictions)
        rows.push([
          key,
          result.last_observation,
          p.datetime,
          p.step,
          p.flux,
          p.alarm,
          result.baseline_flux,
          result.timezone,
          result.historical,
        ]);
    }
    const quote = (x) => `"${String(x ?? "").replaceAll('"', '""')}"`;
    const blob = new Blob(
      ["\uFEFF", rows.map((row) => row.map(quote).join(",")).join("\r\n")],
      { type: "text/csv;charset=utf-8" },
    );
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "p10_forecast_results.csv";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function init() {
    $("today-label").textContent = new Intl.DateTimeFormat("zh-CN", {
      year: "numeric",
      month: "long",
      day: "numeric",
    }).format(new Date());
    $("mode-auto").addEventListener("click", () => setMode("auto"));
    $("mode-direct").addEventListener("click", () => setMode("direct"));
    $("csv-input").addEventListener("input", (event) =>
      setCSV(event.target.value),
    );
    $("file-input").addEventListener("change", (event) =>
      readFile(event.target.files[0]),
    );
    const drop = $("drop-zone");
    for (const event of ["dragenter", "dragover"])
      drop.addEventListener(event, (e) => {
        e.preventDefault();
        drop.classList.add("dragging");
      });
    for (const event of ["dragleave", "drop"])
      drop.addEventListener(event, (e) => {
        e.preventDefault();
        drop.classList.remove("dragging");
      });
    drop.addEventListener("drop", (e) => readFile(e.dataTransfer.files[0]));
    $("predict-button").addEventListener("click", predict);
    $("demo-button").addEventListener("click", demo);
    $("export-button").addEventListener("click", exportCSV);
    try {
      const data = await getJSON("/api/status");
      const count = data.models.filter((item) => item.available).length;
      $("model-count").textContent = String(count).padStart(2, "0");
      $("sidebar-model-count").textContent =
        `${count} / ${data.models.length} 个模型可用`;
    } catch {
      $("sidebar-model-count").textContent = "服务暂不可用";
    }
  }
  init();
})();
