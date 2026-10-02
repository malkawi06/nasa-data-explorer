// NASA Data Explorer - browser build. Python (Pyodide) runs nasa_explorer locally;
// nothing is uploaded anywhere.
import { PROVIDERS, complete, isReady, loadSettings, modelOf, saveSettings } from "./ai.js";

const PYODIDE_VERSION = "314.0.7";
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const MAX_MB = 600; // wasm32 memory is limited; larger files are better run with the CLI

// Always-needed Pyodide packages and pure-Python wheels from PyPI.
const CORE = {
  pkgs: ["micropip", "numpy", "pandas", "xarray", "scipy", "matplotlib", "jinja2", "h5py", "netcdf4", "cftime", "pillow", "xlrd", "requests"],
  pip: ["pymannkendall", "h5netcdf", "openpyxl", "markdown"],
};
// Optional readers: module name used in the reader's `requires` -> what provides it.
const MODULE_SOURCES = {
  rasterio: { pkgs: ["rasterio", "pyproj"], pip: ["rioxarray"] },
  rioxarray: { pkgs: ["rasterio", "pyproj"], pip: ["rioxarray"] },
  geopandas: { pkgs: ["geopandas", "fiona", "shapely", "pyproj"], pip: [] },
  astropy: { pkgs: ["astropy"], pip: [] },
  pyarrow: { pkgs: ["pyarrow"], pip: [] },
  zarr: { pkgs: ["zarr", "numcodecs"], pip: [] },
  bs4: { pkgs: ["beautifulsoup4"], pip: [] },
  docx: { pkgs: ["lxml"], pip: ["python-docx"] },
  pypdf: { pkgs: [], pip: ["pypdf"] },
};
// Preload by extension so the first pass usually succeeds.
const BY_EXT = {
  tif: ["rasterio"], tiff: ["rasterio"], cog: ["rasterio"], jp2: ["rasterio"], img: ["rasterio"],
  shp: ["geopandas"], geojson: ["geopandas"], kml: ["geopandas"], kmz: ["geopandas"], gpkg: ["geopandas"], gml: ["geopandas"], fgb: ["geopandas"],
  fits: ["astropy"], fit: ["astropy"], fts: ["astropy"],
  parquet: ["pyarrow"], pq: ["pyarrow"],
  html: ["bs4"], htm: ["bs4"], docx: ["docx"], pdf: ["pypdf"], zarr: ["zarr"],
};

const T = {
  en: {
    loading: "Loading the Python runtime (first visit downloads ~40 MB)…",
    installing: "Installing analysis libraries…",
    ready: "Ready. Drop files to analyse them.",
    failed: "Could not start Python in this browser: ",
    working: (n) => `Analysing ${n} file(s)…`,
    extra: (m) => `Loading extra libraries: ${m}…`,
    done: (n) => `Done: ${n} report(s).`,
    tooBig: (name) => `${name} is larger than ${MAX_MB} MB. Use the desktop CLI for big files.`,
    badBbox: "Bounding box must be four numbers: W,S,E,N",
    reader: "Reader", kind: "Kind", browser: "In browser", category: "Category", extensions: "Extensions",
    yes: "browser", no: "desktop",
    tabReport: "Report", tabAI: "🤖 AI analysis", tabAsk: "Ask this file",
    runAI: "Run AI analysis", rerunAI: "Run again",
    aiIntro: "The AI reads only the computed summary and statistics (never your file). Every number it writes is checked against those statistics, or against the cited page for papers.",
    needKey: "Choose a provider and paste an API key in “AI settings” first (Gemini and Groq have free tiers).",
    aiStep: (x) => `${x.label}: sending ~${x.tokens.toLocaleString()} tokens to ${x.provider}…`,
    aiWait: (sec) => `Rate limited by the provider; retrying in ${sec}s…`,
    aiDone: (v) => `Done: ${v.verified || 0} verified, ${v.mismatch || 0} wrong, ${v.unsupported || 0} not verified.`,
    aiFailed: "AI failed: ",
    askPlaceholder: "e.g. Which months are warmest? Is there a trend?", askSend: "Ask",
    saved: "Saved.", keyNote: "Your key stays in this browser and is sent only to the provider you choose.",
    desktop: "This format (GRIB / HDF4) needs native libraries the browser lacks. Run: pip install \"nasa-data-explorer[grib,hdf4]\" then nasa-explore FILE",
  },
  ar: {
    title: "مستكشف بيانات ناسا",
    subtitle: "اسحب أي ملف بيانات أو ورقة علمية واحصل على تقرير خلال ثوانٍ. التحليل يعمل داخل متصفحك، فملفاتك لا تغادر جهازك.",
    loading: "جارٍ تحميل بيئة Python (الزيارة الأولى تنزّل نحو 40 ميغابايت)…",
    installing: "جارٍ تثبيت مكتبات التحليل…",
    ready: "جاهز. اسحب الملفات لتحليلها.",
    failed: "تعذّر تشغيل Python في هذا المتصفح: ",
    working: (n) => `جارٍ تحليل ${n} ملف…`,
    extra: (m) => `تحميل مكتبات إضافية: ${m}…`,
    done: (n) => `تم: ${n} تقرير.`,
    tooBig: (name) => `${name} أكبر من ${MAX_MB} ميغابايت. استخدم نسخة سطر الأوامر للملفات الكبيرة.`,
    badBbox: "النطاق الجغرافي يجب أن يكون أربعة أرقام: W,S,E,N",
    drop: "اسحب الملفات هنا أو انقر للاختيار",
    dropHint: "NetCDF وHDF5 وGeoTIFF وShapefile (اختر كل أجزائه) وGeoJSON وFITS وCSV وExcel وJSON وParquet وPDF وDOCX والصور وZIP…",
    options: "خيارات: المتغير، المنطقة، التواريخ",
    var: "المتغير / العمود", bbox: "النطاق W,S,E,N", start: "تاريخ البداية", end: "تاريخ النهاية",
    formats: "الصيغ المدعومة",
    formatsHint: "تُحمَّل المكتبات الإضافية عند الحاجة أول مرة. الصيغ المعلّمة «سطح المكتب» تحتاج نسخة Python المثبّتة (انظر GitHub).",
    footer: "صُنع لتحدي ناسا Space Apps 2026 · غير تابع لناسا",
    open: "فتح", html: "HTML", json: "JSON",
    reader: "القارئ", kind: "النوع", browser: "في المتصفح", category: "الفئة", extensions: "الامتدادات",
    yes: "المتصفح", no: "سطح المكتب",
    tabReport: "التقرير", tabAI: "🤖 تحليل الذكاء الاصطناعي", tabAsk: "اسأل الملف",
    runAI: "شغّل تحليل الذكاء الاصطناعي", rerunAI: "أعد التشغيل",
    aiIntro: "يقرأ الذكاء الاصطناعي الملخص والإحصاءات المحسوبة فقط (وليس ملفك). كل رقم يكتبه يُفحص مقابل هذه الإحصاءات، أو مقابل الصفحة المذكورة في الأوراق العلمية.",
    needKey: "اختر مزوّداً والصق مفتاح API في «إعدادات الذكاء الاصطناعي» أولاً (Gemini وGroq لديهما خطط مجانية).",
    aiStep: (x) => `${x.label}: إرسال نحو ${x.tokens.toLocaleString()} رمز إلى ${x.provider}…`,
    aiWait: (sec) => `تجاوزنا حدّ المزوّد؛ إعادة المحاولة بعد ${sec} ث…`,
    aiDone: (v) => `تم: ${v.verified || 0} مؤكَّد، ${v.mismatch || 0} خاطئ، ${v.unsupported || 0} غير مؤكَّد.`,
    aiFailed: "فشل الذكاء الاصطناعي: ",
    askPlaceholder: "مثلاً: ما أدفأ الأشهر؟ هل يوجد اتجاه؟", askSend: "اسأل",
    saved: "تم الحفظ.", keyNote: "يبقى مفتاحك في هذا المتصفح ويُرسَل فقط إلى المزوّد الذي تختاره.",
    aiSettings: "إعدادات الذكاء الاصطناعي", provider: "المزوّد", model: "النموذج (اختياري)", apiKey: "مفتاح API",
    remember: "تذكّر المفتاح على هذا الجهاز", ollamaHost: "عنوان Ollama", save: "حفظ", getKey: "احصل على مفتاح",
    desktop: "هذه الصيغة (GRIB / HDF4) تحتاج مكتبات غير متوفرة في المتصفح. شغّل: pip install \"nasa-data-explorer[grib,hdf4]\" ثم nasa-explore FILE",
  },
};

const $ = (sel) => document.querySelector(sel);
const state = { lang: "en", py: null, bridge: null, loaded: new Set(), busy: false, runs: 0, status: "loading", statusArg: null };
const t = (key, arg) => {
  const v = T[state.lang][key] ?? T.en[key];
  return typeof v === "function" ? v(arg) : v;
};

function setStatus(key, kind = "busy", arg = null, extra = "") {
  state.status = key; state.statusArg = arg;
  $("#status-dot").className = `dot ${kind}`;
  $("#status-text").textContent = (key ? t(key, arg) : "") + extra;
}

function applyLang(lang) {
  state.lang = lang;
  document.documentElement.lang = lang;
  document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";
  for (const el of document.querySelectorAll("[data-i18n]")) {
    if (!el.dataset.en) el.dataset.en = el.textContent;
    const v = T[lang][el.dataset.i18n];
    el.textContent = typeof v === "string" ? v : el.dataset.en;
  }
  for (const b of document.querySelectorAll(".lang button")) b.setAttribute("aria-pressed", String(b.dataset.lang === lang));
  if (state.status && !$("#status-dot").classList.contains("error")) setStatus(state.status, $("#status-dot").classList[1], state.statusArg);
  if (state.bridge) renderFormats();
}

async function install({ pkgs = [], pip = [] }) {
  const newPkgs = pkgs.filter((p) => !state.loaded.has(p));
  const newPip = pip.filter((p) => !state.loaded.has(p));
  if (newPkgs.length) await state.py.loadPackage(newPkgs);
  if (newPip.length) await state.py.runPythonAsync(`import micropip\nawait micropip.install(${JSON.stringify(newPip)})`);
  for (const p of [...newPkgs, ...newPip]) state.loaded.add(p);
}

async function provide(modules) {
  const todo = modules.filter((m) => MODULE_SOURCES[m] && !state.loaded.has(`mod:${m}`));
  if (!todo.length) return false;
  setStatus("extra", "busy", todo.join(", "));
  for (const m of todo) {
    await install(MODULE_SOURCES[m]);
    state.loaded.add(`mod:${m}`);
  }
  await state.py.runPythonAsync("import importlib; importlib.invalidate_caches()");
  return true;
}

async function boot() {
  try {
    const { loadPyodide } = await import(`${PYODIDE_URL}pyodide.mjs`);
    state.py = await loadPyodide({ indexURL: PYODIDE_URL });
    setStatus("installing");
    await install(CORE);
    const zip = await (await fetch("nasa_explorer.zip", { cache: "no-cache" })).arrayBuffer();
    state.py.unpackArchive(zip, "zip", { extractDir: "/home/pyodide/app" });
    await state.py.runPythonAsync(`
import sys, warnings, matplotlib
sys.path.insert(0, "/home/pyodide/app")
warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import bridge`);
    state.bridge = state.py.pyimport("bridge");
    renderFormats();
    setStatus("ready", "ready");
    $("#drop").classList.remove("disabled");
  } catch (err) {
    console.error(err);
    setStatus("failed", "error", null, String(err.message || err));
  }
}

function renderFormats() {
  const rows = JSON.parse(state.bridge.formats());
  const desktopOnly = new Set(["grib", "hdf4", "pdf"]);
  const head = `<tr><th>${t("category")}</th><th>${t("reader")}</th><th>${t("extensions")}</th><th>${t("browser")}</th></tr>`;
  const body = rows.map((r) => {
    const ok = !desktopOnly.has(r.name);
    return `<tr><td>${esc(r.category)}</td><td>${esc(r.name)}</td><td class="ext">${esc(r.extensions.join(" "))}</td>` +
      `<td><span class="tag ${ok ? "yes" : "no"}">${ok ? t("yes") : t("no")}</span></td></tr>`;
  }).join("");
  $("#formats").innerHTML = head + body;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function readOptions() {
  const bboxText = $("#opt-bbox").value.trim();
  let bbox = null;
  $("#opt-error").textContent = "";
  if (bboxText) {
    bbox = bboxText.split(",").map((v) => Number(v.trim()));
    if (bbox.length !== 4 || bbox.some((v) => !Number.isFinite(v)) || bbox[1] > bbox[3]) {
      $("#opt-error").textContent = t("badBbox");
      $(".options").open = true;
      return null;
    }
  }
  return { var: $("#opt-var").value.trim(), bbox, start: $("#opt-start").value, end: $("#opt-end").value, lang: state.lang };
}

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function fitFrame(frame) {
  frame.addEventListener("load", () => {
    try { frame.style.height = `${frame.contentDocument.documentElement.scrollHeight + 8}px`; } catch { /* cross-origin */ }
  });
}

function renderCard(r) {
  const card = $("#card-tpl").content.firstElementChild.cloneNode(true);
  for (const el of card.querySelectorAll("[data-i18n]")) el.textContent = t(el.dataset.i18n) ?? el.textContent;
  card.querySelector(".file").textContent = r.file;
  card.querySelector(".meta").textContent = `${r.reader} · ${r.kind}`;
  card.querySelector(".headline").textContent = r.headline;
  const desktopOnly = r.missing.some((m) => ["cfgrib", "pyhdf"].includes(m));
  if (r.error || desktopOnly) {
    const err = card.querySelector(".err");
    err.hidden = false;
    err.textContent = r.error || t("desktop");
  }
  const base = r.file.replace(/[^\w.-]+/g, "_");
  card.querySelector(".preview").srcdoc = r.html;
  card.querySelector(".open").onclick = () => {
    const url = URL.createObjectURL(new Blob([r.html], { type: "text/html" }));
    window.open(url, "_blank", "noopener");
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  };
  card.querySelector(".dl-html").onclick = () => download(`${base}.html`, r.html, "text/html");
  card.querySelector(".dl-json").onclick = () => download(`${base}.json`, r.json, "application/json");

  // tabs: Report | AI analysis | Ask
  const tabs = [...card.querySelectorAll("[role=tab]")];
  const select = (name) => {
    for (const tab of tabs) {
      const on = tab.dataset.tab === name;
      tab.setAttribute("aria-selected", String(on));
      tab.tabIndex = on ? 0 : -1;
      card.querySelector(`[data-panel="${tab.dataset.tab}"]`).hidden = !on;
    }
  };
  for (const tab of tabs) tab.addEventListener("click", () => select(tab.dataset.tab));
  card.querySelector(".tablist").addEventListener("keydown", (e) => {
    const i = tabs.findIndex((x) => x.getAttribute("aria-selected") === "true");
    const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key] * (document.documentElement.dir === "rtl" ? -1 : 1);
    if (!step) return;
    const next = tabs[(i + step + tabs.length) % tabs.length];
    select(next.dataset.tab);
    next.focus();
  });
  if (r.error || r.kind === "binary") card.querySelector(".tablist").hidden = true;
  card.querySelector(".ask-input").placeholder = t("askPlaceholder");
  const aiFrame = card.querySelector(".ai-frame");
  fitFrame(aiFrame);
  card.querySelector(".run-ai").onclick = (e) => runAI(card, r, e.currentTarget);
  card.querySelector(".ask-form").onsubmit = (e) => {
    e.preventDefault();
    askFile(card, r);
  };
  return card;
}

function aiSettingsOrPrompt(statusEl) {
  const s = loadSettings();
  if (isReady(s)) return s;
  statusEl.textContent = t("needKey");
  const panel = $("#ai-settings");
  panel.open = true;
  panel.scrollIntoView({ behavior: "smooth", block: "center" });
  $("#ai-key").focus();
  return null;
}

async function drive(start, s, statusEl) {
  let res = JSON.parse(start());
  while (!res.done) {
    statusEl.textContent = t("aiStep", { ...res.step, provider: PROVIDERS[s.provider].label });
    const reply = await complete(s, res.step, (sec) => { statusEl.textContent = t("aiWait", sec); });
    res = JSON.parse(state.bridge.ai_next(res.job, reply));
  }
  return res;
}

async function runAI(card, r, button) {
  const statusEl = card.querySelector(".ai-status");
  const s = aiSettingsOrPrompt(statusEl);
  if (!s) return;
  button.disabled = true;
  try {
    const res = await drive(() => state.bridge.ai_start(r.key, state.lang, s.provider, modelOf(s)), s, statusEl);
    r.html = res.html;
    r.json = res.json;
    card.querySelector(".preview").srcdoc = r.html;
    const frame = card.querySelector(".ai-frame");
    frame.hidden = false;
    frame.srcdoc = res.panel;
    statusEl.textContent = t("aiDone", res.verification || {});
    button.textContent = t("rerunAI");
  } catch (err) {
    console.error(err);
    statusEl.textContent = t("aiFailed") + (err.message || err);
  } finally {
    button.disabled = false;
  }
}

async function askFile(card, r) {
  const input = card.querySelector(".ask-input");
  const log = card.querySelector(".chat-log");
  const question = input.value.trim();
  const statusEl = card.querySelector(".ask-status");
  if (!question) return;
  const s = aiSettingsOrPrompt(statusEl);
  if (!s) return;
  const item = document.createElement("div");
  item.className = "qa";
  item.innerHTML = `<p class="q">${esc(question)}</p><p class="a">…</p>`;
  log.prepend(item);
  input.value = "";
  try {
    const res = await drive(() => state.bridge.chat_start(r.key, question, state.lang), s, statusEl);
    item.querySelector(".a").textContent = res.answer;
    statusEl.textContent = "";
  } catch (err) {
    item.querySelector(".a").textContent = t("aiFailed") + (err.message || err);
  }
}

function initSettings() {
  const s = loadSettings();
  const sel = $("#ai-provider");
  sel.innerHTML = Object.entries(PROVIDERS).map(([k, p]) => `<option value="${k}">${esc(p.label)}</option>`).join("");
  sel.value = s.provider;
  $("#ai-model").value = s.model || "";
  $("#ai-key").value = s.key || "";
  $("#ai-host").value = s.host || "http://localhost:11434";
  $("#ai-remember").checked = Boolean(s.remember);
  const sync = () => {
    const p = PROVIDERS[sel.value];
    $("#ai-model").placeholder = p.model;
    $("#ai-key-row").hidden = !p.key;
    $("#ai-host-row").hidden = p.key;
    $("#ai-get-key").href = p.keyUrl;
  };
  sel.addEventListener("change", sync);
  sync();
  $("#ai-form").addEventListener("submit", (e) => {
    e.preventDefault();
    saveSettings({ provider: sel.value, model: $("#ai-model").value.trim(), key: $("#ai-key").value.trim(),
      host: $("#ai-host").value.trim() || "http://localhost:11434", remember: $("#ai-remember").checked });
    $("#ai-saved").textContent = t("saved");
    setTimeout(() => { $("#ai-saved").textContent = ""; }, 2500);
  });
}

async function analyse(files) {
  if (state.busy || !state.bridge || !files.length) return;
  const options = readOptions();
  if (!options) return;
  for (const f of files) {
    if (f.size > MAX_MB * 1024 * 1024) {
      setStatus("tooBig", "error", f.name);
      return;
    }
  }
  state.busy = true;
  $("#drop").classList.add("disabled");
  const pending = Object.assign(document.createElement("article"), { className: "card pending" });
  pending.textContent = files.map((f) => f.name).join(", ");
  $("#results").prepend(pending);
  try {
    const exts = files.map((f) => f.name.split(".").pop().toLowerCase());
    await provide([...new Set(exts.flatMap((e) => BY_EXT[e] || []))]);
    setStatus("working", "busy", files.length);
    const dir = state.bridge.new_run(`run${++state.runs}`);
    for (const f of files) state.py.FS.writeFile(`${dir}/${f.name}`, new Uint8Array(await f.arrayBuffer()));
    const opts = JSON.stringify(options);
    let results = JSON.parse(state.bridge.analyse_dir(dir, opts));
    // A reader reported a missing optional library (e.g. a format detected by its bytes): load it and retry once.
    if (await provide([...new Set(results.flatMap((r) => r.missing))])) {
      setStatus("working", "busy", files.length);
      results = JSON.parse(state.bridge.analyse_dir(dir, opts));
    }
    pending.replaceWith(...results.map(renderCard));
    setStatus("done", "ready", results.length);
  } catch (err) {
    console.error(err);
    pending.classList.remove("pending");
    pending.innerHTML = `<p class="err">${esc(err.message || err)}</p>`;
    setStatus("failed", "error", null, String(err.message || err).split("\n").slice(-2).join(" "));
  } finally {
    state.busy = false;
    $("#drop").classList.remove("disabled");
  }
}

const drop = $("#drop");
drop.classList.add("disabled");
$("#file-input").addEventListener("change", (e) => {
  analyse([...e.target.files]);
  e.target.value = "";
});
for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); });
drop.addEventListener("drop", (e) => analyse([...e.dataTransfer.files]));
for (const b of document.querySelectorAll(".lang button")) b.addEventListener("click", () => applyLang(b.dataset.lang));

initSettings();
applyLang(navigator.language?.startsWith("ar") ? "ar" : "en");
setStatus("loading");
boot();
