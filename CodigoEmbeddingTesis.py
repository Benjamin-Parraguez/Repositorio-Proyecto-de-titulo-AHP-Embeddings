# ahp_mixed_v7_noneg.py
# AHP con embeddings + contraste uno-vs-todos + asignación híbrida/topK/soft/hard.
# SIN uso de anclas negativas (capa D). Lee:
#  - clave desde gemini.key
#  - corpus desde corpus.txt
#  - anclas desde anchors.json/.yaml (usa layerA, layerB, layerC_in_scope; ignora layerD_exclusions)
#
# Requisitos: numpy, pandas, (opcional pyyaml si usas YAML).
# Compatible con google-generativeai (embed_content) y google.genai (nuevo SDK).

import re, json, argparse, hashlib, datetime as dt
from typing import List, Dict, Tuple, Optional
import numpy as np, pandas as pd
from pathlib import Path
import importlib.util as _u

# --- Rutas por defecto ---
ROOT = Path(__file__).resolve().parent
KEY_PATH = ROOT / "gemini.key"
CORPUS_PATH = ROOT / "financiamiento.txt"   #cambiarlo al texto a analizar txt
ANCHORS_PATH = ROOT / "anchors_ahp_pln_chile_v11.yaml"   # cámbialo o pásalo por --anchors

# --- Config por defecto (ajustables por CLI) ---
DIM = 1536
TASK = "SEMANTIC_SIMILARITY"
MAX_CHARS = 8000

TAU = 0.20              # temperatura softmax (solo en modos soft/hybrid cuando reparte)
SIM_MIN = 0.32          # umbral de similitud positiva para que un segmento cuente
DOC_EQUALIZE = True     # equalización por documento

BETA_OVA = 0.50         # fuerza del contraste uno-vs-todos
CENTER_ROWS = True      # centrar por fila (quita fondo común)

ASSIGN_MODE = "hybrid"  # "soft" | "hard" | "topk" | "hybrid"
MARGIN = 0.02           # gap top - segundo para decidir "duro"
TOPK = 2                # para modos topk/hybrid cuando reparte
DELTA = 0.03            # incluye j con S[j] >= S_top - DELTA

# Modelos/SDK
MODEL_GENAI = "gemini-embedding-001"       # google.genai
MODEL_GG    = "models/text-embedding-004"  # google-generativeai
PREFER_BACKEND = "gg"  # "gg" | "genai" | None(auto)

# ============== UTILIDADES ==============

def _read_key() -> str:
    if not KEY_PATH.exists(): raise FileNotFoundError("No se encontró gemini.key")
    k = KEY_PATH.read_text(encoding="utf-8").strip()
    if not k: raise ValueError("gemini.key está vacío")
    return k

def _normalize_quotes(s: str) -> str:
    return s.replace("“", '"').replace("”", '"')

def _normalize_text(s: str) -> str:
    return " ".join(_normalize_quotes(s).split())

def _split_sentences(text: str, min_tokens: int = 5) -> List[str]:
    sents = re.split(r'(?<=[\.\!\?;:])\s+', text.strip())
    return [s for s in sents if len(s.split()) >= min_tokens]

def _read_segments_with_doc(p: Path, split_sents: bool=False) -> Tuple[List[str], List[str]]:
    """
    Lee corpus: bloques entre "..." o por párrafos en blanco. Soporta [doc=ID] al inicio.
    Si split_sents=True, divide cada bloque en oraciones.
    """
    if not p.exists(): raise FileNotFoundError("No se encontró corpus.txt")
    raw = _normalize_quotes(p.read_text(encoding="utf-8")).strip()
    quotes = re.findall(r'"([^"]+)"', raw, flags=re.DOTALL)
    bloques = quotes if quotes else re.split(r"\n\s*\n", raw)

    segs, docs, vistos = [], [], set()
    for b in bloques:
        s = " ".join(b.split())
        if not s: continue
        doc = "corpus"
        m = re.match(r"^\[doc=([^\]]+)\]\s*(.+)$", s)
        if m:
            doc, s = m.group(1).strip(), m.group(2).strip()
        pieces = _split_sentences(s) if split_sents else [s]
        for piece in pieces:
            piece = piece[:MAX_CHARS]
            if piece and piece not in vistos:
                segs.append(piece); docs.append(doc); vistos.add(piece)
    if not segs: raise RuntimeError("corpus vacío o mal formateado")
    return segs, docs

def _cos_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T

def _checksum_file(path: Path) -> Optional[str]:
    if not path.exists(): return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""): h.update(chunk)
    return h.hexdigest()

# ---- Carga de ANCLAS (solo POSITIVAS) ----
def _load_anchors_pos(path: Path) -> Dict[str,List[str]]:
    """
    Soporta:
      - YAML con lista 'criteria': [{id, layerA, layerB, layerC_in_scope, ...}, ...]
      - JSON/YAML mapa: { "criterios": {Tecnico: "...|[...]", ...} } o nivel superior {Tecnico: "...|[...]", ...}
    Devuelve dict[criterio] -> list[str] (usa layerA, layerB, layerC_in_scope; ignora exclusiones).
    """
    if not path.exists(): raise FileNotFoundError(f"No se encontró {path.name}")
    txt = path.read_text(encoding="utf-8")
    ext = path.suffix.lower()
    if ext == ".json":
        data = json.loads(txt)
    elif ext in (".yaml",".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError("Instala PyYAML para leer YAML: pip install pyyaml")
        data = yaml.safe_load(txt)
    else:
        raise ValueError("Extensión no soportada. Usa .json, .yaml o .yml")

    if not isinstance(data, dict): raise ValueError("Archivo de anclas inválido.")

    def _norm(s: str) -> str: return _normalize_text(s)

    out: Dict[str,List[str]] = {}

    # Caso lista 'criteria'
    for key in ("criteria","criterios","criterios_macro"):
        if isinstance(data.get(key), list):
            for item in data[key]:
                if not isinstance(item, dict): continue
                cid = item.get("id") or item.get("name") or item.get("criterio")
                if not cid: continue
                P = []
                if isinstance(item.get("layerA"), str) and item["layerA"].strip(): P.append(_norm(item["layerA"]))
                if isinstance(item.get("layerB"), str) and item["layerB"].strip(): P.append(_norm(item["layerB"]))
                cscope = item.get("layerC_in_scope")
                if isinstance(cscope, list) and cscope:
                    xs = [_norm(x) for x in cscope if isinstance(x,str) and x.strip()]
                    if xs: P.append("Señales/alcance: " + "; ".join(xs))
                # dedup
                seen=set(); P2=[]
                for a in P:
                    if a and a not in seen: P2.append(a); seen.add(a)
                if P2: out[str(cid)] = P2
            if out: return out

    # Caso mapas clásicos
    def _ci_get(d: dict, key: str):
        for k in d.keys():
            if str(k).lower() == key: return d[k]
        return None
    anchors_obj = None
    for bucket in ("criterios","anchors","anclas"):
        v = _ci_get(data, bucket)
        if isinstance(v, dict): anchors_obj = v; break
    if anchors_obj is None: anchors_obj = data

    META = {"version","meta","schema","about","descripcion","description","notes","titulo","title","domain"}
    for crit, val in list(anchors_obj.items()):
        if str(crit).lower() in META: continue
        if isinstance(val, dict):
            cands=[]
            for f in ("text","texts","anchors","anclas","descripcion","description"):
                vv = val.get(f)
                if isinstance(vv, str): cands.append(vv)
                elif isinstance(vv, list): cands += [x for x in vv if isinstance(x,str)]
            val = cands
        if isinstance(val, str): vals=[val]
        elif isinstance(val, list): vals=[x for x in val if isinstance(x,str)]
        else: continue
        vals = [_norm(x) for x in vals if x and x.strip()]
        if vals: out[str(crit)] = list(dict.fromkeys(vals))

    if not out: raise ValueError("No se encontraron anclas válidas.")
    return out

# ---- SDK Embeddings ----
_HAS_GENAI = _u.find_spec("google.genai") is not None
_HAS_GG    = _u.find_spec("google.generativeai") is not None
if not (_HAS_GENAI or _HAS_GG):
    raise ImportError("Instala: pip install google-generativeai (y opcionalmente google-genai)")

def _make_client(api_key: str):
    if PREFER_BACKEND == "gg" and _HAS_GG:
        import google.generativeai as genai; genai.configure(api_key=api_key); return ("gg", genai)
    if PREFER_BACKEND == "genai" and _HAS_GENAI:
        from google import genai; return ("genai", genai.Client(api_key=api_key))
    if _HAS_GG:
        import google.generativeai as genai; genai.configure(api_key=api_key); return ("gg", genai)
    from google import genai; return ("genai", genai.Client(api_key=api_key))

def _embed_texts(backend, client, texts: List[str]) -> List[List[float]]:
    def _extract_vals(r):
        if isinstance(r, dict):
            emb = r.get("embedding")
            if isinstance(emb, dict): return emb.get("values")
            return emb
        emb = getattr(r, "embedding", None)
        if hasattr(emb, "values"): return emb.values
        return emb
    if backend == "gg":
        genai = client
        vals = []
        if hasattr(genai, "batch_embed_content"):
            try:
                resp = genai.batch_embed_content(
                    model=MODEL_GG,
                    requests=[{"content": t, "task_type": TASK, "output_dimensionality": DIM} for t in texts],
                )
                for r in resp: vals.append(_extract_vals(r))
                return vals
            except Exception:
                resp = genai.batch_embed_content(
                    model=MODEL_GG,
                    requests=[{"content": t, "task_type": TASK} for t in texts],
                )
                for r in resp: vals.append(_extract_vals(r))
                return vals
        # uno a uno (versiones antiguas)
        for t in texts:
            try:
                r = genai.embed_content(model=MODEL_GG, content=t, task_type=TASK, output_dimensionality=DIM)
            except Exception:
                r = genai.embed_content(model=MODEL_GG, content=t, task_type=TASK)
            vals.append(_extract_vals(r))
        return vals
    else:
        from google.genai import types
        try:
            resp = client.models.embed_content(
                model=MODEL_GENAI,
                contents=texts,
                config=types.EmbedContentConfig(task_type=TASK, output_dimensionality=DIM),
            )
        except Exception:
            resp = client.models.embed_content(
                model=MODEL_GENAI,
                contents=texts,
                config=types.EmbedContentConfig(task_type=TASK),
            )
        return [e.values for e in resp.embeddings]

# ============== LÓGICA PRINCIPAL ==============

def _build_S_from_pos(Vseg: np.ndarray,
                      Vpos: np.ndarray,
                      beta_ova: float,
                      center_rows: bool):
    """
    Vpos: (n_crit, dim) ancla positiva por criterio (promedio).
    S = sims_pos - beta_ova * promedio_otros(sims_pos)  [uno-vs-todos]
    (SIN negativos). Luego centrado por fila si corresponde.
    """
    n_crit = Vpos.shape[0]
    sims_pos = _cos_matrix(Vseg, Vpos)              # (n_seg x n_crit)

    sims_ova = np.zeros_like(sims_pos)
    if n_crit > 1:
        sims_sum = sims_pos.sum(axis=1, keepdims=True)
        sims_ova = (sims_sum - sims_pos) / (n_crit - 1)

    S = sims_pos - beta_ova*sims_ova
    if center_rows:
        S = S - S.mean(axis=1, keepdims=True)
    return S, sims_pos, sims_ova

def _assign_contribs(S: np.ndarray,
                     sims_pos: np.ndarray,
                     sim_min: float,
                     mode: str,
                     tau: float,
                     margin: float,
                     topk: int,
                     delta: float) -> np.ndarray:
    """
    Gating con similitud POS cruda (no centrada): conf_pos >= sim_min
    Asignación sobre S (contrastado + centrado): soft | hard | topk | hybrid.
    """
    contribs = np.zeros_like(S)

    conf_pos = sims_pos.max(axis=1)                 # ¡gating acá!
    mask = (conf_pos >= sim_min)

    for i, ok in enumerate(mask):
        if not ok:
            continue
        row = S[i]
        j_top = int(np.argmax(row))
        j2 = int(np.argsort(-row)[1]) if row.size > 1 else j_top
        gap = row[j_top] - row[j2]

        if mode == "hard":
            if gap >= margin:
                contribs[i, j_top] = 1.0

        elif mode == "soft":
            z = row / max(tau, 1e-6); z -= z.max()
            e = np.exp(z); contribs[i] = e / (e.sum() + 1e-12)

        elif mode == "topk":
            k = min(max(topk,1), row.size)
            idx = np.argpartition(-row, k-1)[:k]
            vals = row[idx].copy()
            keep = vals >= (row[j_top] - delta)
            idx = idx[keep]; vals = vals[keep]
            vals = np.clip(vals, 0, None)
            s = vals.sum()
            if s > 0: contribs[i, idx] = vals / s

        else:  # hybrid
            if gap >= margin:
                contribs[i, j_top] = 1.0
            else:
                k = min(max(topk,1), row.size)
                idx = np.argpartition(-row, k-1)[:k]
                vals = row[idx].copy()
                keep = vals >= (row[j_top] - delta)
                idx = idx[keep]; vals = vals[keep]
                vals = np.clip(vals, 0, None)
                s = vals.sum()
                if s > 0: contribs[i, idx] = vals / s
    return contribs

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--anchors", type=str, default=None, help="Ruta anchors.json/.yaml/.yml")
    p.add_argument("--corpus", type=str, default=None, help="Ruta corpus.txt")
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--sim-min", type=float, default=None)
    p.add_argument("--beta-ova", type=float, default=None)
    p.add_argument("--center", type=int, choices=[0,1], default=None)
    p.add_argument("--assign", choices=["soft","hard","topk","hybrid"], default=None)
    p.add_argument("--margin", type=float, default=None)
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--delta", type=float, default=None)
    p.add_argument("--doc-equalize", type=int, choices=[0,1], default=None)
    p.add_argument("--split-sentences", type=int, choices=[0,1], default=0)
    return p.parse_args()

def main():
    # Args
    args = _parse_args()
    anchors_path = Path(args.anchors).expanduser() if args.anchors else ANCHORS_PATH
    corpus_path  = Path(args.corpus).expanduser()  if args.corpus  else CORPUS_PATH

    # Overrides
    tau = args.tau if args.tau is not None else TAU
    sim_min = args.sim_min if args.sim_min is not None else SIM_MIN
    beta_ova = args.beta_ova if args.beta_ova is not None else BETA_OVA
    center_rows = bool(args.center) if args.center is not None else CENTER_ROWS
    assign_mode = args.assign if args.assign is not None else ASSIGN_MODE
    margin = args.margin if args.margin is not None else MARGIN
    topk = args.topk if args.topk is not None else TOPK
    delta = args.delta if args.delta is not None else DELTA
    doc_equalize = bool(args.doc_equalize) if args.doc_equalize is not None else DOC_EQUALIZE
    split_sents = bool(args.split_sentences)

    # SDK
    api_key = _read_key()
    backend, client = _make_client(api_key)

    # Datos
    segmentos, doc_ids = _read_segments_with_doc(corpus_path, split_sents=split_sents)
    anchors_pos = _load_anchors_pos(anchors_path)
    criterios = list(anchors_pos.keys())

    # Aplana y embebe POS (promedio por criterio)
    pos_texts, slices_pos = [], []
    for c in criterios:
        xs = anchors_pos.get(c, [])
        start = len(pos_texts); pos_texts.extend(xs); end = len(pos_texts)
        slices_pos.append((start, end))
    Vpos_all = np.array(_embed_texts(backend, client, pos_texts), dtype=np.float32)
    Vpos = np.vstack([Vpos_all[i:j].mean(axis=0) for (i,j) in slices_pos])

    # Segmentos
    Vseg = np.array(_embed_texts(backend, client, segmentos), dtype=np.float32)

    # Contruir S (contrastiva sin negativos)
    S, sims_pos, sims_ova = _build_S_from_pos(
        Vseg=Vseg, Vpos=Vpos, beta_ova=beta_ova, center_rows=center_rows
    )

    # Contribuciones por segmento (gating con sims_pos)
    contribs = _assign_contribs(S, sims_pos, sim_min, assign_mode, tau, margin, topk, delta)

    # Equalización por documento
    if doc_equalize and len(set(doc_ids)) > 1:
        doc_ids_np = np.array(doc_ids)
        w_docs = []
        for d in np.unique(doc_ids_np):
            m = (doc_ids_np == d)
            if m.any():
                w_docs.append(contribs[m].mean(axis=0))
        agg = np.vstack(w_docs).sum(axis=0) if w_docs else contribs.sum(axis=0)
    else:
        agg = contribs.sum(axis=0)

    # Pesos
    total = agg.sum()
    if total <= 1e-12:
        w = np.zeros(len(criterios), dtype=np.float32)  # sin aportes
    else:
        w = agg / total

    # Matriz A y CR (formalidad)
    n = len(criterios)
    A = w[:,None] / (w[None,:] + 1e-12)
    ew,_ = np.linalg.eig(A)
    CI = ((max(ew.real)-n)/(n-1)) if n>1 else 0.0
    RI = {1:0.0,2:0.0,3:0.58,4:0.90,5:1.12}.get(n,1.12)
    CR = CI/RI if RI>0 else 0.0

    # CSVs
    df_pesos = pd.DataFrame({"criterio": criterios, "peso_local": w}).sort_values("peso_local", ascending=False)

    top_idx = S.argmax(axis=1)
    second = np.partition(S, -2, axis=1)[:, -2] if S.shape[1] >= 2 else S.max(axis=1)
    top_val = S.max(axis=1)
    gap = top_val - second
    df_seg = pd.DataFrame({
        "doc_id": doc_ids,
        "segmento": segmentos,
        "top_name": [criterios[j] for j in top_idx],
        "top_val": top_val,
        "gap": gap,
        "conf_pos": sims_pos.max(axis=1),
        **{f"sim_pos_{c}": sims_pos[:, j] for j,c in enumerate(criterios)},
        **{f"sim_ova_{c}": sims_ova[:, j] for j,c in enumerate(criterios)},
        **{f"sim_{c}":     S[:, j]        for j,c in enumerate(criterios)},
        **{f"w_{c}":       contribs[:, j] for j,c in enumerate(criterios)},
    })

    df_A = pd.DataFrame(A, index=criterios, columns=criterios)

    df_pesos.to_csv(ROOT/"pesos_macro.csv", index=False)
    df_seg.to_csv(ROOT/"segmentos_clasificados.csv", index=False)
    df_A.to_csv(ROOT/"matriz_pareada_ahp.csv", float_format="%.6f")

    # Metadatos
    meta = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "backend": "google-generativeai" if backend=="gg" else "google-genai",
        "model": MODEL_GG if backend=="gg" else MODEL_GENAI,
        "task_type": TASK,
        "dim_requested": DIM,
        "dim_effective": int(Vseg.shape[1]) if Vseg.ndim==2 else None,
        "tau": tau,
        "sim_min": sim_min,
        "beta_ova": beta_ova,
        "center_rows": center_rows,
        "assign_mode": assign_mode,
        "margin": margin,
        "topk": topk,
        "delta": delta,
        "doc_equalize": doc_equalize,
        "n_segmentos_total": int(len(segmentos)),
        "n_criterios": int(len(criterios)),
        "CR": float(CR),
        "CI": float(CI),
        "RI": float(RI),
        "corpus_path": str(corpus_path),
        "corpus_sha256": _checksum_file(corpus_path),
        "anchors_path": str(anchors_path),
        "anchors_sha256": _checksum_file(anchors_path),
        "archivos_generados": ["pesos_macro.csv", "segmentos_clasificados.csv", "matriz_pareada_ahp.csv"]
    }
    (ROOT/"run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Prints
    print(f"\nBackend: {meta['backend']} | Modelo: {meta['model']}")
    print("=== PESOS LOCALES ===")
    for c,p in zip(df_pesos['criterio'], df_pesos['peso_local']):
        print(f"{c:10s}: {p:.4f}")
    print(f"CR = {meta['CR']:.4f}")
    print("Archivos: pesos_macro.csv, segmentos_clasificados.csv, matriz_pareada_ahp.csv, run_meta.json")

if __name__ == "__main__":
    main()
