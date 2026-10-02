// NASA Data Explorer - browser build. Python (Pyodide) runs nasa_explorer in a Web Worker
// (worker.js); nothing is uploaded anywhere.
import { PROVIDERS, complete, isReady, loadSettings, modelOf, saveSettings } from "./ai.js";

const MAX_MB = 600; // wasm32 memory is limited; larger files are better run with the CLI
// Preload optional libraries by extension so the first pass usually succeeds.
const BY_EXT = {
  tif: ["rasterio"], tiff: ["rasterio"], cog: ["rasterio"], jp2: ["rasterio"], img: ["rasterio"],
  lbl: ["rasterio"], cub: ["rasterio"], hgt: ["rasterio"], bil: ["rasterio"], dem: ["rasterio"], grd: ["rasterio"],
  asc: ["rasterio"], vic: ["rasterio"],
  shp: ["geopandas"], geojson: ["geopandas"], kml: ["geopandas"], kmz: ["geopandas"], gpkg: ["geopandas"], gml: ["geopandas"], fgb: ["geopandas"],
  fits: ["astropy"], fit: ["astropy"], fts: ["astropy"],
  parquet: ["pyarrow"], pq: ["pyarrow"],
  html: ["bs4"], htm: ["bs4"], docx: ["docx"], pdf: ["pypdf"], zarr: ["zarr"],
};
const DESKTOP_ONLY = new Set(["grib", "hdf4", "pdf"]);

const T = {
  en: {
    loading: "Starting…",
    pRuntime: "Downloading Python for your browser (~40 MB, first visit only)…",
    pPackages: (x) => `Installing analysis libraries (${x.done + 1}/${x.total}): ${x.label}`,
    pExtra: (x) => `Loading extra libraries: ${x.label}…`,
    pAnalyse: (x) => `Analysing ${x.label} (${x.done + 1} of ${x.total})…`,
    ready: "Ready. Drop files to analyse them.",
    queued: (n) => `${n} file(s) will be analysed as soon as Python is ready…`,
    failed: "Could not start Python in this browser: ",
    done: (n) => `Done: ${n} report(s).`,
    tooBig: (name) => `${name} is larger than ${MAX_MB} MB. Use the desktop CLI for big files.`,
    badBbox: "Bounding box must be four numbers: W,S,E,N",
    reader: "Reader", browser: "Where", category: "Category", extensions: "Extensions",
    yes: "browser", no: "desktop",
    runAI: "Run AI analysis", rerunAI: "Run again",
    needKey: "Choose a provider and paste an API key in “AI settings” first (Gemini and Groq have free tiers).",
    aiStep: (x) => `${x.label}: sending ~${x.tokens.toLocaleString()} tokens to ${x.provider}…`,
    aiWait: (sec) => `Rate limited by the provider; retrying in ${sec}s…`,
    aiDone: (v) => `Done: ${v.verified || 0} verified, ${v.mismatch || 0} wrong, ${v.unsupported || 0} not verified${v.visual ? `, ${v.visual} visual (not machine-checked)` : ""}.`,
    aiFailed: "AI failed: ",
    askPlaceholder: "e.g. Which months are warmest? Is there a trend?",
    saved: "Saved.",
    desktop: "This format (GRIB / HDF4) needs native libraries the browser lacks. Run: pip install \"nasa-data-explorer[grib,hdf4]\" then nasa-explore FILE",
    analysing: "analysing…",
    powerFetching: "Fetching NASA POWER data…",
    powerFailed: "NASA POWER request failed: ",
    badPoint: "Latitude must be -90..90 and longitude -180..180.",
    analogFetching: (x) => `Fetching NASA POWER climate ${x.done}/${x.total}…`,
    analogRanking: "Scoring sites…",
    analogNoSites: "Add at least one site (name, lat, lon) or keep the built-in list.",
    analogBadLine: (x) => `Line ${x} is not "name, lat, lon".`,
    analogDone: (x) => `Done: ${x.ok} of ${x.total} sites scored.`,
  },
  ar: {
    title: "مستكشف بيانات ناسا",
    loading: "جارٍ البدء…",
    pRuntime: "تنزيل Python للمتصفح (نحو 40 ميغابايت، في الزيارة الأولى فقط)…",
    pPackages: (x) => `تثبيت مكتبات التحليل (${x.done + 1}/${x.total}): ${x.label}`,
    pExtra: (x) => `تحميل مكتبات إضافية: ${x.label}…`,
    pAnalyse: (x) => `تحليل ${x.label} (${x.done + 1} من ${x.total})…`,
    ready: "جاهز. اسحب الملفات لتحليلها.",
    queued: (n) => `سيُحلَّل ${n} ملف حالما يجهز Python…`,
    failed: "تعذّر تشغيل Python في هذا المتصفح: ",
    done: (n) => `تم: ${n} تقرير.`,
    tooBig: (name) => `${name} أكبر من ${MAX_MB} ميغابايت. استخدم نسخة سطر الأوامر للملفات الكبيرة.`,
    badBbox: "النطاق الجغرافي يجب أن يكون أربعة أرقام: W,S,E,N",
    drop: "اسحب الملفات أو انقر للاختيار",
    dropHint: "تبقى ملفاتك على جهازك. اختر كل أجزاء Shapefile معاً.",
    options: "خيارات: المتغير، المنطقة، التواريخ",
    var: "المتغير / العمود", bbox: "النطاق W,S,E,N", start: "تاريخ البداية", end: "تاريخ النهاية",
    formats: "الصيغ المدعومة",
    formatsHint: "تُحمَّل المكتبات الإضافية عند الحاجة أول مرة. صيغ «سطح المكتب» تحتاج نسخة Python المثبّتة.",
    footer: "صُنع لتحدي ناسا Space Apps 2026 · غير تابع لناسا",
    open: "فتح في تبويب جديد", html: "HTML", json: "JSON",
    reader: "القارئ", browser: "أين", category: "الفئة", extensions: "الامتدادات",
    yes: "المتصفح", no: "سطح المكتب",
    tabReport: "التقرير", tabAI: "🤖 تحليل الذكاء الاصطناعي", tabAsk: "اسأل الملف",
    runAI: "شغّل تحليل الذكاء الاصطناعي", rerunAI: "أعد التشغيل",
    aiIntro: "يقرأ الذكاء الاصطناعي الملخص والإحصاءات المحسوبة فقط، وليس ملفك (إلا إذا فعّلت «إرسال الصور» لملف صورة). كل رقم يكتبه يُفحص مقابل هذه الإحصاءات، أو مقابل الصفحة المذكورة في الأوراق العلمية.",
    needKey: "اختر مزوّداً والصق مفتاح API في «إعدادات الذكاء الاصطناعي» أولاً (Gemini وGroq لديهما خطط مجانية).",
    aiStep: (x) => `${x.label}: إرسال نحو ${x.tokens.toLocaleString()} رمز إلى ${x.provider}…`,
    aiWait: (sec) => `تجاوزنا حدّ المزوّد؛ إعادة المحاولة بعد ${sec} ث…`,
    aiDone: (v) => `تم: ${v.verified || 0} مؤكَّد، ${v.mismatch || 0} خاطئ، ${v.unsupported || 0} غير مؤكَّد${v.visual ? `، ${v.visual} بصرية (غير مفحوصة آلياً)` : ""}.`,
    aiFailed: "فشل الذكاء الاصطناعي: ",
    askPlaceholder: "مثلاً: ما أدفأ الأشهر؟ هل يوجد اتجاه؟", askSend: "اسأل",
    saved: "تم الحفظ.", keyNote: "يبقى مفتاحك في هذا المتصفح ويُرسَل فقط إلى المزوّد الذي تختاره.",
    aiSettings: "إعدادات الذكاء الاصطناعي (بمفتاحك الخاص)", provider: "المزوّد", model: "النموذج (اختياري)", apiKey: "مفتاح API",
    remember: "تذكّر المفتاح على هذا الجهاز", sendImages: "أرسل الصور للمزوّد لتحليل بصري (Gemini / Claude فقط؛ الصورة تغادر هذا المتصفح)", ollamaHost: "عنوان Ollama", save: "حفظ", getKey: "احصل على مفتاح مجاني",
    desktop: "هذه الصيغة (GRIB / HDF4) تحتاج مكتبات غير متوفرة في المتصفح. شغّل: pip install \"nasa-data-explorer[grib,hdf4]\" ثم nasa-explore FILE",
    heroTitle: "افهم أي ملف بيانات من ناسا خلال دقائق",
    heroLead: "اسحب ملف NetCDF أو HDF5 أو GeoTIFF أو CSV أو FITS أو Shapefile أو ورقة PDF أو ملف ZIP يجمعها. ستحصل على خرائط واتجاهات وفحوصات جودة وملخص بلغة واضحة.",
    f1t: "خصوصية", f1: "يعمل Python داخل متصفحك. لا يُرفع أي ملف.",
    f2t: "أكثر من 30 صيغة", f2: "تُكتشف الصيغة من محتوى الملف نفسه، حتى لو كان الامتداد خاطئاً.",
    f3t: "ذكاء اصطناعي مُدقَّق", f3: "ملخص اختياري يُفحص فيه كل رقم مقابل البيانات.",
    analysing: "جارٍ التحليل…",
    analogTitle: "🌍 مكتشف المواقع الشبيهة بالقمر / المريخ", analogTarget: "الهدف",
    tMoon: "القمر - قاعدة دائمة عند القطب الجنوبي", tMars: "المريخ - موقع هبوط بخطوط عرض منخفضة/متوسطة", tMarsPolar: "المريخ - موقع قطبي غني بالجليد",
    analogBuiltin: "مواقع شبيهة معروفة + مرشحون في الأردن (24)", analogSites: "مواقعك، سطر لكل موقع: الاسم، خط العرض، خط الطول",
    analogRun: "رتّب المواقع",
    analogHint: "مناخ كل موقع من NASA POWER (2001-2020). ملفات خرائط الارتفاع أو NDVI التي حللتها وتغطي الموقع تضيف التضاريس والنباتات. كل عامل مشروح.",
    powerTitle: "☀️ مناخ NASA POWER لأي نقطة", lat: "خط العرض", lon: "خط الطول", powerRun: "اجلب البيانات وحلّلها",
    powerHint: "حرارة وأمطار ورطوبة ورياح وإشعاع شمسي وغيوم يومية من NASA POWER. اترك التواريخ فارغة لآخر 10 سنوات كاملة.",
    powerFetching: "جارٍ جلب بيانات NASA POWER…", powerFailed: "فشل طلب NASA POWER: ",
    badPoint: "خط العرض بين -90 و90، وخط الطول بين -180 و180.",
    analogFetching: (x) => `جلب مناخ NASA POWER ${x.done}/${x.total}…`, analogRanking: "حساب علامات المواقع…",
    analogNoSites: "أضف موقعاً واحداً على الأقل (الاسم، خط العرض، خط الطول) أو أبقِ القائمة المدمجة.",
    analogBadLine: (x) => `السطر ${x} ليس بالشكل «الاسم، خط العرض، خط الطول».`,
    analogDone: (x) => `تم: حُسبت علامات ${x.ok} من ${x.total} موقع.`,
  },
};

const $ = (sel) => document.querySelector(sel);
const state = {
  lang: "en", ready: false, busy: false, runs: 0, queue: [], formats: [],
  results: new Map(), order: [], selected: null, status: ["loading"],
};
const t = (key, arg) => {
  const v = T[state.lang][key] ?? T.en[key];
  return typeof v === "function" ? v(arg) : v;
};
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// ---------------------------------------------------------------- worker RPC + progress

const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
const pending = new Map();
let nextId = 0;
const rpc = (fn, ...args) => new Promise((resolve, reject) => {
  const id = ++nextId;
  pending.set(id, { resolve, reject });
  const transfer = fn === "analyse" ? args[1].map((f) => f.data) : [];
  worker.postMessage({ id, fn, args }, transfer);
});
worker.onmessage = ({ data }) => {
  if (data.type === "progress") return onProgress(data);
  const p = pending.get(data.id);
  pending.delete(data.id);
  if (data.error !== undefined) p.reject(new Error(data.error));
  else p.resolve(data.result);
};
worker.onerror = (e) => setStatus(["failed"], "error", e.message);

let progressGen = 0;
function setProgress(fraction) {
  const gen = ++progressGen;
  $("#progress").classList.toggle("indeterminate", fraction === null);
  $("#progress-bar").style.width = fraction === null ? "" : `${Math.round(fraction * 100)}%`;
  // hide the bar shortly after completion unless new progress arrived meanwhile
  if (fraction === 1) setTimeout(() => { if (gen === progressGen) $("#progress-bar").style.width = "0"; }, 700);
}

function onProgress(p) {
  if (p.phase === "runtime") { setStatus(["pRuntime"]); setProgress(null); }
  if (p.phase === "packages") { setStatus(["pPackages", p]); setProgress(0.25 + 0.75 * (p.done / p.total)); }
  if (p.phase === "extra") { setStatus(["pExtra", p]); setProgress(null); }
  if (p.phase === "analyse") { setStatus(["pAnalyse", p]); setProgress(p.total ? p.done / p.total : null); }
}

function setStatus(key, kind = "busy", extra = "") {
  state.status = key;
  $("#status-dot").className = `dot ${kind}`;
  $("#status-text").textContent = t(key[0], key[1]) + extra;
}

// ---------------------------------------------------------------- language

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
  $(".ask-input").placeholder = t("askPlaceholder");
  const dot = $("#status-dot").classList;
  if (!dot.contains("error")) setStatus(state.status, dot[1] || "busy");
  renderFormats();
  renderList();
  if (state.selected) renderDetail();
}

// ---------------------------------------------------------------- boot

async function boot() {
  try {
    state.formats = JSON.parse(await rpc("init"));
    state.ready = true;
    renderFormats();
    setProgress(1);
    setStatus(["ready"], "ready");
    if (state.queue.length) analyse(state.queue.splice(0));
  } catch (err) {
    console.error(err);
    setProgress(0);
    setStatus(["failed"], "error", String(err.message || err));
  }
}

function renderFormats() {
  if (!state.formats.length) return;
  const head = `<tr><th>${t("category")}</th><th>${t("reader")}</th><th>${t("extensions")}</th><th>${t("browser")}</th></tr>`;
  const body = state.formats.map((r) => {
    const ok = !DESKTOP_ONLY.has(r.name);
    return `<tr><td>${esc(r.category)}</td><td>${esc(r.name)}</td><td class="ext">${esc(r.extensions.join(" "))}</td>` +
      `<td><span class="tag ${ok ? "yes" : "no"}">${ok ? t("yes") : t("no")}</span></td></tr>`;
  }).join("");
  $("#formats").innerHTML = head + body;
}

// ---------------------------------------------------------------- file list (master)

function stateIcon(r) {
  if (r.pending) return `<span class="spinner" aria-label="${esc(t("analysing"))}"></span>`;
  if (r.error || r.kind === "binary") return `<span class="fi-state bad" aria-hidden="true">!</span>`;
  if (r.issues) return `<span class="fi-state warn" title="${r.issues}" aria-hidden="true">⚠</span>`;
  return `<span class="fi-state ok" aria-hidden="true">✓</span>`;
}

function renderList() {
  const ul = $("#files");
  ul.innerHTML = state.order.map((key) => {
    const r = state.results.get(key);
    const meta = r.pending ? t("analysing") : `${r.reader} · ${r.kind}${r.headline ? " · " + r.headline : ""}`;
    return `<li><button type="button" class="file-item${r.pending ? " pending" : ""}" data-key="${esc(key)}"` +
      ` aria-current="${key === state.selected}"${r.pending ? " disabled" : ""}>` +
      `<span class="fi-name">${esc(r.file)}</span>${stateIcon(r)}<span class="fi-meta">${esc(meta)}</span></button></li>`;
  }).join("");
}

$("#files").addEventListener("click", (e) => {
  const btn = e.target.closest(".file-item");
  if (btn && !btn.disabled) select(btn.dataset.key);
});

// ---------------------------------------------------------------- report rendering (detail)

// Reports are complete HTML pages; render them in a shadow root so their styles stay
// scoped while the page scrolls normally (sticky section bar, no iframe height hacks).
// Report text comes from the user's files. The Python side escapes it; this second line of
// defence drops anything that could still run code (the page's CSP blocks it too).
function sanitize(node) {
  node.querySelectorAll("script,iframe,object,embed,form,base,meta,link").forEach((el) => el.remove());
  for (const el of node.querySelectorAll("*")) {
    for (const { name, value } of [...el.attributes]) {
      const url = /^(href|src|xlink:href|action|formaction)$/i.test(name);
      if (/^on/i.test(name) || (url && /^\s*(javascript|vbscript):/i.test(value))) el.removeAttribute(name);
    }
  }
}

function renderInto(host, html) {
  const doc = new DOMParser().parseFromString(html, "text/html");
  sanitize(doc.body);
  const css = [...doc.querySelectorAll("style")].map((s) => s.textContent).join("\n")
    .replace(/:root/g, ":host").replace(/(^|[}\s])body\s*\{/g, "$1:host{display:block;");
  const root = host.shadowRoot || host.attachShadow({ mode: "open" });
  // the page header already shows the file name, so the report's own title is hidden here
  root.innerHTML = `<style>${css}\n:host main{max-width:none;padding:0}\n.rep-head h1{display:none}` +
    `\n.toc{top:calc(var(--top,54px) + 3px)}</style>` +
    `<main dir="${doc.documentElement.dir || "ltr"}">${doc.querySelector("main")?.innerHTML || doc.body.innerHTML}</main>`;
}

function shadowAnchors(host) {
  host.addEventListener("click", (e) => {
    const a = e.composedPath().find((n) => n.tagName === "A");
    const href = a?.getAttribute("href") || "";
    if (!href.startsWith("#")) return;
    e.preventDefault();
    host.shadowRoot.getElementById(href.slice(1))?.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}
shadowAnchors($(".preview"));

function select(key) {
  state.selected = key;
  renderList();
  renderDetail();
  selectTab("report");
  if (window.matchMedia("(max-width: 900px)").matches) $("#detail").scrollIntoView({ behavior: "smooth" });
}

function renderDetail() {
  const r = state.results.get(state.selected);
  if (!r) return;
  $("#empty").hidden = true;
  $("#detail").hidden = false;
  $("#detail .file").textContent = r.file;
  $("#detail .meta").textContent = `${r.reader} · ${r.kind}`;
  const desktopOnly = r.missing.some((m) => ["cfgrib", "pyhdf"].includes(m));
  const err = $("#detail .err");
  err.hidden = !(r.error || desktopOnly);
  err.textContent = r.error || (desktopOnly ? t("desktop") : "");
  $("#detail .tablist").hidden = Boolean(r.error || r.kind === "binary");
  renderInto($(".preview"), r.html);
  const aiHost = $(".ai-frame");
  aiHost.hidden = !r.ai.panel;
  if (r.ai.panel) renderInto(aiHost, r.ai.panel);
  $(".ai-status").textContent = r.ai.status || "";
  $(".run-ai").textContent = r.ai.panel ? t("rerunAI") : t("runAI");
  $(".run-ai").disabled = Boolean(r.ai.running);
  $(".chat-log").innerHTML = r.chat.map((c) => `<div class="qa"><p class="q">${esc(c.q)}</p><p class="a">${esc(c.a)}</p></div>`).join("");
  $(".ask-status").textContent = "";
}

const tabs = [...document.querySelectorAll("[role=tab]")];
function selectTab(name) {
  for (const tab of tabs) {
    const on = tab.dataset.tab === name;
    tab.setAttribute("aria-selected", String(on));
    tab.tabIndex = on ? 0 : -1;
    document.querySelector(`[data-panel="${tab.dataset.tab}"]`).hidden = !on;
  }
}
for (const tab of tabs) tab.addEventListener("click", () => selectTab(tab.dataset.tab));
$(".tablist").addEventListener("keydown", (e) => {
  const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key] * (document.documentElement.dir === "rtl" ? -1 : 1);
  if (!step) return;
  const i = tabs.findIndex((x) => x.getAttribute("aria-selected") === "true");
  const next = tabs[(i + step + tabs.length) % tabs.length];
  selectTab(next.dataset.tab);
  next.focus();
});

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
const current = () => state.results.get(state.selected);
const baseName = (r) => r.file.replace(/[^\w.-]+/g, "_");
$(".open").onclick = () => {
  const url = URL.createObjectURL(new Blob([current().html], { type: "text/html" }));
  window.open(url, "_blank", "noopener");
  setTimeout(() => URL.revokeObjectURL(url), 60000);
};
$(".dl-html").onclick = () => download(`${baseName(current())}.html`, current().html, "text/html");
$(".dl-json").onclick = () => download(`${baseName(current())}.json`, current().json, "application/json");

// ---------------------------------------------------------------- analysis

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

function enqueue(files) {
  files = files.filter(Boolean);
  if (!files.length) return;
  const big = files.find((f) => f.size > MAX_MB * 1024 * 1024);
  if (big) return setStatus(["tooBig", big.name], "error");
  if (!state.ready || state.busy) {
    state.queue.push(...files);
    if (!state.ready) setStatus(["queued", state.queue.length]);
    return;
  }
  analyse(files);
}

async function analyse(files) {
  const options = readOptions();
  if (!options) return;
  state.busy = true;
  const run = `run${++state.runs}`;
  const placeholder = `${run}/…`;
  state.results.set(placeholder, { file: files.map((f) => f.name).join(", "), pending: true });
  state.order.unshift(placeholder);
  renderList();
  try {
    const exts = files.map((f) => f.name.split(".").pop().toLowerCase());
    await rpc("provide", [...new Set(exts.flatMap((e) => BY_EXT[e] || []))]);
    const payload = await Promise.all(files.map(async (f) => ({ name: f.name, data: await f.arrayBuffer() })));
    const opts = JSON.stringify(options);
    let results = JSON.parse(await rpc("analyse", run, payload, opts));
    // a reader reported a missing optional library (format found by its bytes): load it and retry once
    if (await rpc("provide", [...new Set(results.flatMap((r) => r.missing))])) {
      const again = await Promise.all(files.map(async (f) => ({ name: f.name, data: await f.arrayBuffer() })));
      results = JSON.parse(await rpc("analyse", run, again, opts));
    }
    state.results.delete(placeholder);
    state.order = state.order.filter((k) => k !== placeholder);
    for (const r of results.reverse()) {
      const quality = JSON.parse(r.json).analysis?.quality || [];
      r.issues = quality.filter((q) => q.level !== "info").length;
      r.ai = {};
      r.chat = [];
      state.results.set(r.key, r);
      state.order.unshift(r.key);
    }
    setProgress(1);
    setStatus(["done", results.length], "ready");
    select(results[results.length - 1].key);
  } catch (err) {
    console.error(err);
    state.results.set(placeholder, { file: files.map((f) => f.name).join(", "), error: String(err.message || err),
      kind: "binary", reader: "error", headline: "", missing: [], html: "", json: "{}", ai: {}, chat: [] });
    renderList();
    setProgress(0);
    setStatus(["failed"], "error", String(err.message || err).split("\n").slice(-2).join(" "));
  } finally {
    state.busy = false;
    if (state.queue.length) analyse(state.queue.splice(0));
  }
}

const drop = $("#drop");
$("#file-input").addEventListener("change", (e) => {
  enqueue([...e.target.files]);
  e.target.value = "";
});
for (const target of [drop, $("#view")]) {
  for (const ev of ["dragenter", "dragover"]) target.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
  for (const ev of ["dragleave", "drop"]) target.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); });
  target.addEventListener("drop", (e) => enqueue([...e.dataTransfer.files]));
}
for (const b of document.querySelectorAll(".lang button")) b.addEventListener("click", () => applyLang(b.dataset.lang));

// ---------------------------------------------------------------- NASA POWER + analog finder

function point(latEl, lonEl, statusEl) {
  const lat = Number(latEl.value), lon = Number(lonEl.value);
  if (latEl.value === "" || lonEl.value === "" || !(Math.abs(lat) <= 90) || !(Math.abs(lon) <= 180)) {
    statusEl.textContent = t("badPoint");
    return null;
  }
  return [lat, lon];
}

async function fetchText(url) {
  const resp = await fetch(url);
  const text = await resp.text();
  if (!resp.ok) throw new Error(`${resp.status} ${text.slice(0, 200)}`);
  return text;
}

$("#power-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const status = $("#power-status");
  const p = point($("#power-lat"), $("#power-lon"), status);
  if (!p) return;
  if (!state.ready) { status.textContent = t("loading"); return; }
  status.textContent = t("powerFetching");
  try {
    const urls = JSON.parse(await rpc("power_urls", p[0], p[1], $("#power-start").value, $("#power-end").value));
    const csv = await fetchText(urls.daily);
    if (csv.trimStart().startsWith("{")) throw new Error(csv.slice(0, 300)); // POWER returns JSON errors
    const name = `POWER_daily_${p[0]}_${p[1]}_${urls.start}_${urls.end}.csv`.replace(/-/g, "");
    status.textContent = "";
    enqueue([new File([csv], name, { type: "text/csv" })]);
  } catch (err) {
    status.textContent = t("powerFailed") + (err.message || err);
  }
});

const CLIM = "nde-power-clim:";
async function climatology(site) {
  const id = `${CLIM}${Number(site.lat).toFixed(3)},${Number(site.lon).toFixed(3)}`;
  try { const hit = localStorage.getItem(id); if (hit) return JSON.parse(hit); } catch { /* storage blocked */ }
  const urls = JSON.parse(await rpc("power_urls", site.lat, site.lon));
  const data = JSON.parse(await fetchText(urls.climatology));
  try { localStorage.setItem(id, JSON.stringify(data)); } catch { /* quota: skip cache */ }
  return data;
}

function parseSites(text, statusEl) {
  const sites = [];
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
  for (let i = 0; i < lines.length; i++) {
    const parts = lines[i].split(",").map((x) => x.trim());
    const lat = Number(parts.at(-2)), lon = Number(parts.at(-1));
    if (parts.length < 3 || !Number.isFinite(lat) || !Number.isFinite(lon) || Math.abs(lat) > 90 || Math.abs(lon) > 180) {
      statusEl.textContent = t("analogBadLine", i + 1);
      return null;
    }
    sites.push({ name: parts.slice(0, -2).join(", "), lat, lon, analog_for: "" });
  }
  return sites;
}

$("#analog-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const status = $("#analog-status");
  if (!state.ready) { status.textContent = t("loading"); return; }
  const own = parseSites($("#analog-sites").value, status);
  if (!own) return;
  const builtin = $("#analog-builtin").checked ? JSON.parse(await rpc("analog_sites")).sites : [];
  const sites = [...builtin, ...own];
  if (!sites.length) { status.textContent = t("analogNoSites"); return; }
  const clims = new Array(sites.length).fill(null);
  let done = 0;
  const pull = async (queue) => {
    for (let i = queue.shift(); i !== undefined; i = queue.shift()) {
      try { clims[i] = await climatology(sites[i]); } catch (err) { console.warn(sites[i].name, err); }
      status.textContent = t("analogFetching", { done: ++done, total: sites.length });
    }
  };
  const queue = sites.map((_, i) => i);
  await Promise.all([pull(queue), pull(queue), pull(queue)]); // 3 requests at a time
  status.textContent = t("analogRanking");
  const run = `run${++state.runs}`;
  try {
    const results = JSON.parse(await rpc("analog_run", run, $("#analog-target").value, JSON.stringify(sites), JSON.stringify(clims), state.lang));
    for (const r of results) {
      r.issues = 0;
      r.ai = {};
      r.chat = [];
      state.results.set(r.key, r);
      state.order.unshift(r.key);
    }
    status.textContent = t("analogDone", { ok: clims.filter(Boolean).length, total: sites.length });
    select(results[0].key);
  } catch (err) {
    status.textContent = String(err.message || err);
  }
});

// ---------------------------------------------------------------- AI (bring your own key)

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

async function drive(startFn, s, onStatus) {
  let res = JSON.parse(await startFn());
  while (!res.done) {
    onStatus(t("aiStep", { ...res.step, provider: PROVIDERS[s.provider].label }));
    const reply = await complete(s, res.step, (sec) => onStatus(t("aiWait", sec)));
    res = JSON.parse(await rpc("ai_next", res.job, reply));
  }
  return res;
}

$(".run-ai").onclick = async () => {
  const r = current();
  const s = aiSettingsOrPrompt($(".ai-status"));
  if (!s || !r) return;
  const show = (msg) => { r.ai.status = msg; if (current() === r) $(".ai-status").textContent = msg; };
  r.ai.running = true;
  $(".run-ai").disabled = true;
  try {
    const res = await drive(() => rpc("ai_start", r.key, state.lang, s.provider, modelOf(s), Boolean(s.sendImages)), s, show);
    r.html = res.html;
    r.json = res.json;
    r.ai.panel = res.panel;
    show(t("aiDone", res.verification || {}));
  } catch (err) {
    console.error(err);
    show(t("aiFailed") + (err.message || err));
  } finally {
    r.ai.running = false;
    if (current() === r) renderDetail();
  }
};

$(".ask-form").onsubmit = async (e) => {
  e.preventDefault();
  const r = current();
  const input = $(".ask-input");
  const question = input.value.trim();
  if (!question || !r) return;
  const s = aiSettingsOrPrompt($(".ask-status"));
  if (!s) return;
  const entry = { q: question, a: "…" };
  r.chat.unshift(entry);
  input.value = "";
  renderDetail();
  try {
    const res = await drive(() => rpc("chat_start", r.key, question, state.lang), s,
      (msg) => { if (current() === r) $(".ask-status").textContent = msg; });
    entry.a = res.answer;
  } catch (err) {
    entry.a = t("aiFailed") + (err.message || err);
  }
  if (current() === r) renderDetail();
};

function initSettings() {
  const s = loadSettings();
  const sel = $("#ai-provider");
  sel.innerHTML = Object.entries(PROVIDERS).map(([k, p]) => `<option value="${k}">${esc(p.label)}</option>`).join("");
  sel.value = s.provider;
  $("#ai-model").value = s.model || "";
  $("#ai-key").value = s.key || "";
  $("#ai-host").value = s.host || "http://localhost:11434";
  $("#ai-remember").checked = Boolean(s.remember);
  $("#ai-images").checked = Boolean(s.sendImages);
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
      host: $("#ai-host").value.trim() || "http://localhost:11434", remember: $("#ai-remember").checked,
      sendImages: $("#ai-images").checked });
    $("#ai-saved").textContent = t("saved");
    setTimeout(() => { $("#ai-saved").textContent = ""; }, 2500);
  });
}

initSettings();
applyLang(navigator.language?.startsWith("ar") ? "ar" : "en");
setStatus(["loading"]);
setProgress(null);
boot();
