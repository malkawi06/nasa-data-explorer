// Bring-your-own-key AI calls made directly from the browser to the chosen provider.
// The key never reaches any server of ours (there is none). Only prompts built by
// web/bridge.py are sent: computed summaries and statistics, never the uploaded file.

export const PROVIDERS = {
  gemini: { label: "Google Gemini (free tier)", model: "gemini-2.5-flash", key: true, keyUrl: "https://aistudio.google.com/apikey" },
  groq: { label: "Groq (free tier)", model: "llama-3.3-70b-versatile", key: true, keyUrl: "https://console.groq.com/keys" },
  anthropic: { label: "Anthropic Claude", model: "claude-sonnet-5-5", key: true, keyUrl: "https://console.anthropic.com/settings/keys" },
  ollama: { label: "Ollama (local, no key)", model: "qwen2.5:7b", key: false, keyUrl: "https://ollama.com/download" },
};

const STORE = "nde-ai-settings";
const CACHE = "nde-ai-cache:";
const MAX_RETRIES = 5;

function storage(kind) {
  try {
    const s = window[kind];
    s.setItem("__t", "1");
    s.removeItem("__t");
    return s;
  } catch {
    return null; // private mode / blocked storage: settings live only in memory
  }
}

let memory = null;

export function loadSettings() {
  for (const kind of ["localStorage", "sessionStorage"]) {
    const raw = storage(kind)?.getItem(STORE);
    if (raw) {
      try { return { ...JSON.parse(raw), remember: kind === "localStorage" }; } catch { /* ignore corrupt entry */ }
    }
  }
  return memory ?? { provider: "gemini", model: "", key: "", host: "http://localhost:11434", remember: false };
}

export function saveSettings(settings) {
  memory = settings;
  const data = JSON.stringify({ ...settings, remember: undefined });
  storage("localStorage")?.removeItem(STORE);
  storage("sessionStorage")?.removeItem(STORE);
  storage(settings.remember ? "localStorage" : "sessionStorage")?.setItem(STORE, data);
}

export function isReady(s) {
  return Boolean(PROVIDERS[s.provider]) && (!PROVIDERS[s.provider].key || s.key.trim().length > 8);
}

export const modelOf = (s) => s.model.trim() || PROVIDERS[s.provider].model;

async function sha256(text) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

class ProviderError extends Error {
  constructor(message, status, retryAfter) {
    super(message);
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

function request(s, { system, prompt, json }) {
  const model = modelOf(s);
  switch (s.provider) {
    case "gemini":
      return {
        url: `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`,
        headers: { "content-type": "application/json", "x-goog-api-key": s.key.trim() },
        body: {
          systemInstruction: { parts: [{ text: system }] },
          contents: [{ role: "user", parts: [{ text: prompt }] }],
          generationConfig: { temperature: 0.2, ...(json ? { responseMimeType: "application/json" } : {}) },
        },
        read: (d) => (d.candidates?.[0]?.content?.parts || []).map((p) => p.text || "").join(""),
      };
    case "groq":
      return {
        url: "https://api.groq.com/openai/v1/chat/completions",
        headers: { "content-type": "application/json", authorization: `Bearer ${s.key.trim()}` },
        body: {
          model, temperature: 0.2,
          messages: [{ role: "system", content: system }, { role: "user", content: prompt }],
          ...(json ? { response_format: { type: "json_object" } } : {}),
        },
        read: (d) => d.choices?.[0]?.message?.content || "",
      };
    case "anthropic":
      return {
        url: "https://api.anthropic.com/v1/messages",
        headers: {
          "content-type": "application/json",
          "x-api-key": s.key.trim(),
          "anthropic-version": "2023-06-01",
          "anthropic-dangerous-direct-browser-access": "true",
        },
        body: { model, max_tokens: 4096, system, messages: [{ role: "user", content: prompt }] },
        read: (d) => (d.content || []).filter((b) => b.type === "text").map((b) => b.text).join(""),
      };
    case "ollama":
      return {
        url: `${s.host.replace(/\/+$/, "")}/api/chat`,
        headers: { "content-type": "application/json" },
        body: {
          model, stream: false, options: { temperature: 0.2 }, ...(json ? { format: "json" } : {}),
          messages: [{ role: "system", content: system }, { role: "user", content: prompt }],
        },
        read: (d) => d.message?.content || "",
      };
    default:
      throw new ProviderError(`Unknown provider ${s.provider}`);
  }
}

async function callOnce(s, step) {
  const req = request(s, step);
  let resp;
  try {
    resp = await fetch(req.url, { method: "POST", headers: req.headers, body: JSON.stringify(req.body) });
  } catch {
    const hint = s.provider === "ollama"
      ? ` Start Ollama with OLLAMA_ORIGINS=${location.origin} so this page may call it.`
      : " Check your connection or try another provider.";
    throw new ProviderError(`Could not reach ${PROVIDERS[s.provider].label}.${hint}`, 0);
  }
  if (resp.ok) return req.read(await resp.json());
  const detail = (await resp.text()).slice(0, 300);
  const retryAfter = Number(resp.headers.get("retry-after")) || null;
  if (resp.status === 401 || resp.status === 403) throw new ProviderError(`The API key was rejected (${resp.status}).`, resp.status);
  throw new ProviderError(`${PROVIDERS[s.provider].label} error ${resp.status}: ${detail}`, resp.status, retryAfter);
}

/** One model call with an on-device cache and backoff for free-tier rate limits (429). */
export async function complete(s, step, onWait = () => {}) {
  const id = CACHE + (await sha256([s.provider, modelOf(s), step.json, step.system, step.prompt].join("|")));
  const cached = storage("localStorage")?.getItem(id);
  if (cached !== null && cached !== undefined) return cached;
  let delay = 2;
  for (let attempt = 1; ; attempt++) {
    try {
      const text = await callOnce(s, step);
      try { storage("localStorage")?.setItem(id, text); } catch { /* quota full: skip caching */ }
      return text;
    } catch (err) {
      const retryable = err.status === 429 || err.status >= 500;
      if (!retryable || attempt >= MAX_RETRIES) throw err;
      const wait = err.retryAfter || delay + Math.random();
      onWait(Math.ceil(wait));
      await new Promise((r) => setTimeout(r, wait * 1000));
      delay = Math.min(delay * 2, 60);
    }
  }
}
