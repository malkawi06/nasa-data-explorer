// NASA Data Explorer - browser build. Python (Pyodide) runs nasa_explorer locally;
// nothing is uploaded anywhere.
const PYODIDE_VERSION = "314.0.7";
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const MAX_MB = 600; // wasm32 memory is limited; larger files are better run with the CLI

// Always-needed Pyodide packages and pure-Python wheels from PyPI.
const CORE = {
  pkgs: ["micropip", "numpy", "pandas", "xarray", "scipy", "matplotlib", "jinja2", "h5py", "netcdf4", "cftime", "pillow", "xlrd"],
  pip: ["pymannkendall", "h5netcdf", "openpyxl", "markdown"],
};
// Optional readers: module name used in the reader's `requires` -> what provides it.
const PROVIDERS = {
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
  const todo = modules.filter((m) => PROVIDERS[m] && !state.loaded.has(`mod:${m}`));
  if (!todo.length) return false;
  setStatus("extra", "busy", todo.join(", "));
  for (const m of todo) {
    await install(PROVIDERS[m]);
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
  return card;
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

applyLang(navigator.language?.startsWith("ar") ? "ar" : "en");
setStatus("loading");
boot();
