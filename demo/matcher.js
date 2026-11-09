// Browser port of app/analysis.py. Every function mirrors the Python one it
// names, so the demo and the API give the same answer for the same text.

// app/preprocessing.py clean_text: lower-case, drop everything that is not a
// word character or whitespace.
export function cleanText(text) {
  return text.toLowerCase().replace(/[^\p{L}\p{N}_\s]/gu, "");
}

// scikit-learn TfidfVectorizer defaults: token_pattern \b\w\w+\b, raw counts,
// smooth idf = ln((1 + n) / (1 + df)) + 1, l2-normalised rows.
function tokens(text) {
  return text.match(/[\p{L}\p{N}_]{2,}/gu) ?? [];
}

function counts(list) {
  const m = new Map();
  for (const t of list) m.set(t, (m.get(t) ?? 0) + 1);
  return m;
}

function tfidfVectors(docs) {
  const tfs = docs.map((d) => counts(tokens(d)));
  const df = new Map();
  for (const tf of tfs) for (const t of tf.keys()) df.set(t, (df.get(t) ?? 0) + 1);
  const n = docs.length;
  return tfs.map((tf) => {
    const v = new Map();
    let norm = 0;
    for (const [t, c] of tf) {
      const w = c * (Math.log((1 + n) / (1 + df.get(t))) + 1);
      v.set(t, w);
      norm += w * w;
    }
    norm = Math.sqrt(norm) || 1;
    for (const [t, w] of v) v.set(t, w / norm);
    return v;
  });
}

// app/model.py tfidf_similarity, plus the shared terms ranked by how much
// each one adds to the cosine (the "why" behind the keyword score).
export function tfidfSimilarity(a, b) {
  const [va, vb] = tfidfVectors([a, b]);
  let score = 0;
  const shared = [];
  for (const [t, w] of va) {
    if (vb.has(t)) {
      const c = w * vb.get(t);
      score += c;
      shared.push({ term: t, contribution: c });
    }
  }
  shared.sort((x, y) => y.contribution - x.contribution);
  return { score, shared };
}

export function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// app/bias.py bias_check: whole-word tokens only, so "the" never counts as "he".
export const MALE_WORDS = ["he", "his", "man"];
export const FEMALE_WORDS = ["she", "her", "woman"];

export function biasTokens(text) {
  return text.toLowerCase().match(/[\p{L}\p{N}_']+/gu) ?? [];
}

export function biasCheck(text) {
  const set = new Set(biasTokens(text));
  const male = MALE_WORDS.filter((w) => set.has(w)).sort();
  const female = FEMALE_WORDS.filter((w) => set.has(w)).sort();
  const dominant = male.length > female.length ? "male" : female.length > male.length ? "female" : "neutral";
  return {
    male_bias: male.length,
    female_bias: female.length,
    dominant,
    flagged_words: [...male, ...female],
  };
}

// app/model.py split_chunks: lines / sentences, cleaned, empty ones dropped.
export function splitChunks(text) {
  return text.split(/[\n.;!?]+/).map((c) => cleanText(c).trim()).filter(Boolean);
}

// app/model.py mentions: the cleaned skill name appears as whole words.
export function mentions(cleanedText, skill) {
  const words = cleanText(skill).split(/\s+/).filter(Boolean);
  // Cleaned words hold only letters, digits and "_", so they need no escaping.
  const body = words.join("\\s+");
  return new RegExp(`(?<![\\p{L}\\p{N}_])${body}(?![\\p{L}\\p{N}_])`, "u").test(cleanedText);
}

// app/model.py semantic_skill_match: named in the text, or some line /
// sentence is within cosine `threshold` of the skill name.
function skillsIn(rawText, skills, skillEmbs, chunkEmbs, threshold) {
  const cleaned = cleanText(rawText);
  return new Set(skills.filter((s, i) =>
    mentions(cleaned, s) || chunkEmbs.some((c) => cosine(c, skillEmbs[i]) > threshold)));
}

// app/model.py explain
function explain(matched, missing) {
  if (!matched.length && !missing.length) return "The job description names none of the tracked skills.";
  if (missing.length === 0) return "Your CV covers all major required skills.";
  if (matched.length > missing.length) return "You match most requirements but can improve in some areas.";
  return "Your CV is missing several key skills for this role.";
}

// app/analysis.py analyze_texts. `embed(texts)` returns one vector per text
// (all-MiniLM-L6-v2, mean pooled, 256-token limit, like sentence-transformers).
export async function analyzeTexts(cvRaw, jdRaw, skills, embed, skillThreshold = 0.4) {
  skills = [...skills].sort();
  const cv = cleanText(cvRaw);
  const jd = cleanText(jdRaw);
  const cvChunks = splitChunks(cvRaw);
  const jdChunks = splitChunks(jdRaw);

  const [cvEmb, jdEmb] = await embed([cv, jd]);
  const skillEmbs = await embed(skills);
  const cvChunkEmbs = cvChunks.length ? await embed(cvChunks) : [];
  const jdChunkEmbs = jdChunks.length ? await embed(jdChunks) : [];

  const sbert = cosine(cvEmb, jdEmb);
  const tfidf = tfidfSimilarity(cv, jd);

  const cvSkills = skillsIn(cvRaw, skills, skillEmbs, cvChunkEmbs, skillThreshold);
  const jdSkills = skillsIn(jdRaw, skills, skillEmbs, jdChunkEmbs, skillThreshold);
  const matched = [...jdSkills].filter((s) => cvSkills.has(s)).sort();
  const missing = [...jdSkills].filter((s) => !cvSkills.has(s)).sort();

  const skillScore = matched.length / (matched.length + missing.length + 1e-5);
  const advanced = 0.5 * sbert + 0.2 * tfidf.score + 0.3 * skillScore;

  return {
    sbert_score: sbert,
    tfidf_score: tfidf.score,
    advanced_score: advanced,
    matched_skills: matched,
    missing_skills: missing,
    bias: biasCheck(jd),
    explanation: explain(matched, missing),
    shared_terms: tfidf.shared,
  };
}
