import { analyzeTexts, cleanText } from "./matcher.js";

const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1/dist/transformers.min.js";
const PDFJS_URL = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/6.4.299/pdf.min.mjs";
const PDFJS_WORKER_URL = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/6.4.299/pdf.worker.min.mjs";
const MODEL_ID = "Xenova/all-MiniLM-L6-v2";
const MAX_TOKENS = 256; // sentence-transformers max_seq_length for this model

const EXAMPLE_CV = `Nadia Haddad — Machine Learning Engineer

Summary
Machine learning engineer with 4 years of experience taking models from notebook to production. I build NLP and forecasting systems in Python and deploy them as small, well-tested services.

Experience
ML Engineer, retail analytics company (2022–now)
- Built a demand-forecasting model in PyTorch that cut stock-outs by 18% across 40 stores.
- Fine-tuned a sentence-transformer for product search; served it with FastAPI in Docker on Kubernetes.
- Set up CI/CD in GitHub Actions with unit tests and model checks before every release.

Data Analyst, logistics startup (2020–2022)
- Wrote SQL and pandas pipelines for weekly delivery KPIs and dashboards.
- Trained scikit-learn models to predict late deliveries.

Skills
Python, PyTorch, scikit-learn, pandas, SQL, FastAPI, Docker, Kubernetes, Git, NLP, sentence embeddings`;

const EXAMPLE_JD = `NLP Engineer (Kuwait City or remote)

We are looking for an NLP engineer to join the data team. He will design and ship text classification and semantic search features used by thousands of customers every day.

What you will do
- Train and evaluate deep learning models for Arabic and English text with PyTorch or TensorFlow.
- Deploy models as APIs with Docker and Kubernetes, and own their monitoring.
- Work with the analytics team on SQL pipelines and data quality.

What we are looking for
- 3+ years of Python and machine learning in production.
- Experience with NLP, transformers and embeddings.
- Familiarity with computer vision is a plus.`;

const $ = (id) => document.getElementById(id);
const els = {
  cv: $("cv"), jd: $("jd"), pdf: $("pdf"), run: $("run"), clear: $("clear"), status: $("status"),
  progress: $("progress"), results: $("results"), theme: $("theme"),
};

// ---------- theme ----------
function storage(fn) { try { return fn(window.localStorage); } catch { return null; } }
function applyTheme(t) {
  if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
  else delete document.documentElement.dataset.theme;
}
applyTheme(storage((s) => s.getItem("theme")));
els.theme.addEventListener("click", () => {
  const current = document.documentElement.dataset.theme
    || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = current === "dark" ? "light" : "dark";
  applyTheme(next);
  storage((s) => s.setItem("theme", next));
});

// ---------- inputs ----------
function count(el, out) {
  const words = (el.value.match(/\S+/g) || []).length;
  $(out).textContent = `${words} words`;
}
els.cv.value = EXAMPLE_CV;
els.jd.value = EXAMPLE_JD;
for (const [el, out] of [[els.cv, "cv-count"], [els.jd, "jd-count"]]) {
  count(el, out);
  el.addEventListener("input", () => count(el, out));
}
els.clear.addEventListener("click", () => {
  els.cv.value = ""; els.jd.value = "";
  count(els.cv, "cv-count"); count(els.jd, "jd-count");
  els.results.classList.remove("show");
  els.cv.focus();
});

function setStatus(text, isError = false) {
  els.status.textContent = text;
  els.status.classList.toggle("err", isError);
}

els.pdf.addEventListener("change", async () => {
  const file = els.pdf.files[0];
  if (!file) return;
  setStatus(`Reading ${file.name}…`);
  try {
    const pdfjs = await import(PDFJS_URL);
    pdfjs.GlobalWorkerOptions.workerSrc = PDFJS_WORKER_URL;
    const doc = await pdfjs.getDocument({ data: await file.arrayBuffer() }).promise;
    const pages = [];
    for (let i = 1; i <= doc.numPages; i++) {
      const content = await (await doc.getPage(i)).getTextContent();
      pages.push(content.items.map((it) => it.str + (it.hasEOL ? "\n" : " ")).join(""));
    }
    const text = pages.join("\n").trim();
    if (!text) throw new Error("no text layer (scanned PDF?)");
    els.cv.value = text;
    count(els.cv, "cv-count");
    setStatus(`Loaded ${doc.numPages} page${doc.numPages > 1 ? "s" : ""} from ${file.name}.`);
  } catch (err) {
    setStatus(`Could not read that PDF: ${err.message}. Paste the CV text instead.`, true);
  } finally {
    els.pdf.value = "";
  }
});

// ---------- model ----------
let embedPromise = null;
function loadEmbedder() {
  if (embedPromise) return embedPromise;
  embedPromise = (async () => {
    const { AutoTokenizer, AutoModel, mean_pooling } = await import(TRANSFORMERS_URL);
    const bar = els.progress.querySelector("i");
    const files = new Map();
    els.progress.style.display = "block";
    const progress_callback = (e) => {
      if (e.status === "progress" && e.total) {
        files.set(e.file, [e.loaded, e.total]);
        let loaded = 0, total = 0;
        for (const [l, t] of files.values()) { loaded += l; total += t; }
        bar.style.width = `${(100 * loaded) / total}%`;
        setStatus(`Downloading the model… ${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`);
      }
    };
    const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, { progress_callback });
    const model = await AutoModel.from_pretrained(MODEL_ID, { dtype: "fp32", progress_callback });
    els.progress.style.display = "none";
    return async (texts) => {
      const inputs = tokenizer(texts, { padding: true, truncation: true, max_length: MAX_TOKENS });
      const { last_hidden_state } = await model(inputs);
      return mean_pooling(last_hidden_state, inputs.attention_mask).tolist();
    };
  })();
  embedPromise.catch(() => { embedPromise = null; els.progress.style.display = "none"; });
  return embedPromise;
}

let skillsPromise = null;
function loadSkills() {
  skillsPromise ??= fetch("skills.json").then((r) => {
    if (!r.ok) throw new Error(`skills.json: HTTP ${r.status}`);
    return r.json();
  });
  return skillsPromise;
}

// ---------- render ----------
const STOP_WORDS = new Set(("a an and are as at be by for from has have in is it its of on or our the their this " +
  "to was we were will with you your who what which all any can into more than that them they these those i my me " +
  "us also but not so if do does use used using every day years year").split(" "));
const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function chips(el, items, cls, emptyText) {
  el.innerHTML = items.length
    ? items.map((s) => `<span class="chip ${cls}">${esc(s)}</span>`).join("")
    : `<span class="empty">${esc(emptyText)}</span>`;
}

function highlight(text, shared, biasWords = new Set()) {
  return text.split(/([\p{L}\p{N}_']+)/u).map((part, i) => {
    if (i % 2 === 0) return esc(part);
    const word = cleanText(part);
    if (biasWords.has(part.toLowerCase())) return `<mark class="bias">${esc(part)}</mark>`;
    if (shared.has(word)) return `<mark>${esc(part)}</mark>`;
    return esc(part);
  }).join("");
}

function bar(id, value) {
  const v = Math.max(0, Math.min(1, value));
  $(`b-${id}`).style.width = `${(v * 100).toFixed(1)}%`;
  $(`n-${id}`).textContent = value.toFixed(3);
}

function render(r, cvText, jdText) {
  $("overall").innerHTML = `${Math.round(r.advanced_score * 100)}<span>%</span>`;
  $("explanation").textContent = r.explanation;
  bar("sbert", r.sbert_score);
  bar("tfidf", r.tfidf_score);
  const required = r.matched_skills.length + r.missing_skills.length;
  bar("skill", required ? r.matched_skills.length / required : 0);
  if (!required) $("n-skill").textContent = "–";

  chips($("matched"), r.matched_skills, "ok", required ? "None of the required skills found." : "The job names none of the 16 tracked skills.");
  chips($("missing"), r.missing_skills, "miss", required ? "Nothing missing." : "Nothing to compare.");

  const flagged = r.bias.flagged_words;
  $("bias").innerHTML = flagged.length
    ? `<p style="margin:0 0 8px">Found ${flagged.map((w) => `<span class="flag-word">${esc(w)}</span>`).join(", ")} (leans ${esc(r.bias.dominant)}).</p>
       <p class="sub" style="margin:0">Consider "they" or "the engineer" so the ad reads as open to everyone.</p>`
    : `<p class="empty" style="margin:0">None found. Words like "the" or "shell" are not counted.</p>`;

  // The score counts every shared word; the chips and highlights skip filler
  // words so the meaningful overlap is easy to see.
  const meaningful = r.shared_terms.filter((t) => !STOP_WORDS.has(t.term));
  const shared = new Set(meaningful.map((t) => t.term));
  chips($("terms"), meaningful.slice(0, 16).map((t) => t.term), "term", "No meaningful words in common.");
  $("hl-cv").innerHTML = highlight(cvText, shared);
  $("hl-jd").innerHTML = highlight(jdText, shared, new Set(flagged));

  els.results.classList.add("show");
}

// ---------- run ----------
async function run() {
  const cvText = els.cv.value.trim();
  const jdText = els.jd.value.trim();
  if (!cvText || !jdText) {
    setStatus("Add both a CV and a job description, then compare.", true);
    return;
  }
  els.run.disabled = true;
  setStatus("Loading the model…");
  try {
    const [embed, skills] = await Promise.all([loadEmbedder(), loadSkills()]);
    setStatus("Comparing…");
    const t0 = performance.now();
    const r = await analyzeTexts(cvText, jdText, skills, embed);
    render(r, cvText, jdText);
    window.__lastResult = r; // handy for checking the numbers against the Python API
    setStatus(`Done in ${Math.round(performance.now() - t0)} ms, all in your browser.`);
    if (!window.__autorun) els.results.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  } catch (err) {
    console.error(err);
    setStatus(`Something went wrong: ${err.message}. Check your connection and try again.`, true);
  } finally {
    els.run.disabled = false;
  }
}
els.run.addEventListener("click", run);

// ?run=1 compares the example straight away (used for the README screenshot).
if (new URLSearchParams(location.search).has("run")) { window.__autorun = true; run(); }
