// Runs Pyodide + nasa_explorer off the main thread so the page never freezes.
// Protocol: main -> {id, fn, args}; worker -> {id, result} | {id, error} | {type:"progress", ...}.

const PYODIDE_VERSION = "314.0.7";
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

// Always-needed Pyodide packages, loaded in small batches so progress can be reported.
const CORE_BATCHES = [
  ["micropip", "numpy", "requests"],
  ["pandas", "cftime"],
  ["scipy"],
  ["xarray", "h5py"],
  ["matplotlib", "pillow"],
  ["jinja2", "xlrd"],
];
const CORE_PIP = ["pymannkendall", "h5netcdf", "openpyxl", "markdown"];

// Optional readers: module name in a reader's `requires` -> what provides it.
const MODULE_SOURCES = {
  rasterio: { pkgs: ["rasterio"], pip: ["rioxarray"] },
  rioxarray: { pkgs: ["rasterio"], pip: ["rioxarray"] },
  geopandas: { pkgs: ["geopandas", "fiona", "shapely"], pip: [] },
  astropy: { pkgs: ["astropy"], pip: [] },
  pyarrow: { pkgs: ["pyarrow"], pip: [] },
  zarr: { pkgs: ["zarr", "numcodecs"], pip: [] },
  bs4: { pkgs: ["beautifulsoup4"], pip: [] },
  docx: { pkgs: ["lxml"], pip: ["python-docx"] },
  pypdf: { pkgs: [], pip: ["pypdf"] },
};
// PROJ must load before GDAL-based packages: the other order makes the first pyproj call
// kill the runtime ("null function or function signature mismatch").
const NEEDS_PROJ_FIRST = new Set(["fiona", "rasterio", "geopandas"]);

let py = null;
let bridge = null;
const loaded = new Set();
const progress = (data) => postMessage({ type: "progress", ...data });

async function loadPkgs(pkgs) {
  const todo = pkgs.filter((p) => !loaded.has(p));
  if (todo.some((p) => NEEDS_PROJ_FIRST.has(p)) && !loaded.has("pyproj")) {
    await py.loadPackage(["pyproj"]);
    loaded.add("pyproj");
  }
  if (todo.length) await py.loadPackage(todo);
  todo.forEach((p) => loaded.add(p));
}

async function pipInstall(names) {
  const todo = names.filter((p) => !loaded.has(p));
  if (!todo.length) return;
  await py.runPythonAsync(`import micropip\nawait micropip.install(${JSON.stringify(todo)})`);
  todo.forEach((p) => loaded.add(p));
}

const api = {
  async init() {
    progress({ phase: "runtime", done: 0, total: 1 });
    const { loadPyodide } = await import(`${PYODIDE_URL}pyodide.mjs`);
    py = await loadPyodide({ indexURL: PYODIDE_URL });
    const steps = CORE_BATCHES.length + 2;
    for (let i = 0; i < CORE_BATCHES.length; i++) {
      progress({ phase: "packages", done: i, total: steps, label: CORE_BATCHES[i].join(", ") });
      await loadPkgs(CORE_BATCHES[i]);
    }
    progress({ phase: "packages", done: CORE_BATCHES.length, total: steps, label: CORE_PIP.join(", ") });
    await pipInstall(CORE_PIP);
    progress({ phase: "packages", done: steps - 1, total: steps, label: "nasa_explorer" });
    const zip = await (await fetch("nasa_explorer.zip", { cache: "no-cache" })).arrayBuffer();
    py.unpackArchive(zip, "zip", { extractDir: "/home/pyodide/app" });
    await py.runPythonAsync(`
import sys, warnings, matplotlib
sys.path.insert(0, "/home/pyodide/app")
warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import bridge`);
    bridge = py.pyimport("bridge");
    return bridge.formats();
  },

  async provide(modules) {
    const todo = modules.filter((m) => MODULE_SOURCES[m] && !loaded.has(`mod:${m}`));
    if (!todo.length) return false;
    progress({ phase: "extra", label: todo.join(", ") });
    for (const m of todo) {
      await loadPkgs(MODULE_SOURCES[m].pkgs);
      await pipInstall(MODULE_SOURCES[m].pip);
      loaded.add(`mod:${m}`);
    }
    await py.runPythonAsync("import importlib; importlib.invalidate_caches()");
    return true;
  },

  async analyse(run, files, options) {
    const dir = bridge.new_run(run);
    for (const f of files) py.FS.writeFile(`${dir}/${f.name}`, new Uint8Array(f.data));
    const onFile = (label, i, n) => progress({ phase: "analyse", label, done: i, total: n });
    return bridge.analyse_dir(dir, options, onFile);
  },

  // AI and chat steps (the provider call itself happens on the main thread, see ai.js)
  ai_start: (...a) => bridge.ai_start(...a),
  ai_next: (...a) => bridge.ai_next(...a),
  chat_start: (...a) => bridge.chat_start(...a),
};

onmessage = async ({ data: { id, fn, args } }) => {
  try {
    const result = await api[fn](...(args || []));
    postMessage({ id, result });
  } catch (err) {
    postMessage({ id, error: String(err?.message || err) });
  }
};
