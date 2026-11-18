#!/usr/bin/env python3
"""
SBIR/STTR biomedical asset-map pipeline
=======================================

Cleans the SBIR.gov award export, keeps the most recent five award years,
separates biomedical awards from everything else, groups awards into topics,
and writes the data behind the interactive page (index.html).

Usage
-----
    # Baseline that runs anywhere (scikit-learn only):
    python sbir_pipeline.py --csv SBIR_award_data.csv --embedder tfidf

    # SPECTER2 embeddings (needs internet access to huggingface.co; GPU advised):
    pip install torch transformers adapters
    python sbir_pipeline.py --csv SBIR_award_data.csv --embedder specter2

Both embedders feed the same classifier, the same clustering and the same
evaluation, so the two quality reports (quality_report.json) are directly
comparable.

Outputs (in --out, default ./build)
-----------------------------------
    awards_clean_5yr.csv        every award in the window, cleaned, with labels
    biomedical_awards.csv       the biomedical subset
    teams.csv                   company + PI teams and what they have won
    quality_report.json         how well the biomedical separation worked
    topics.json                 topic terms and sizes (edit topic_labels.json to rename)
    data.json                   compact data consumed by the page
    index.html                  the page (only if dashboard_template.html is present)

Requirements: pandas, numpy, scikit-learn, zipcodes (pip install zipcodes)
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (normalized_mutual_info_score, precision_score,
                             recall_score, roc_auc_score, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import normalize

SEED = 42

# --------------------------------------------------------------------------
# 1. Cleaning
# --------------------------------------------------------------------------
AGENCY_SHORT = {
    "Department of Defense": "DoD",
    "Department of Health and Human Services": "HHS",
    "National Aeronautics and Space Administration": "NASA",
    "Department of Energy": "DOE",
    "National Science Foundation": "NSF",
    "Department of Agriculture": "USDA",
    "Environmental Protection Agency": "EPA",
    "Department of Commerce": "DOC",
    "Department of Education": "ED",
    "Department of Transportation": "DOT",
    "Department of Homeland Security": "DHS",
    "Nuclear Regulatory Commission": "NRC",
    "Department of Interior": "DOI",
}

# NIH institute/centre codes as they appear in grant numbers (R43 NS 132653)
NIH_IC = {
    "AI": "NIAID", "HL": "NHLBI", "CA": "NCI", "AG": "NIA", "DA": "NIDA",
    "HD": "NICHD", "AR": "NIAMS", "EY": "NEI", "EB": "NIBIB", "NS": "NINDS",
    "DC": "NIDCD", "DE": "NIDCR", "ES": "NIEHS", "DK": "NIDDK", "TR": "NCATS",
    "MD": "NIMHD", "HG": "NHGRI", "OD": "NIH OD", "AT": "NCCIH", "GM": "NIGMS",
    "LM": "NLM", "MH": "NIMH", "AA": "NIAAA", "NR": "NINR", "RR": "NCRR",
    "CK": "CDC", "OH": "CDC", "IP": "CDC", "CE": "CDC", "DP": "CDC",
    "EH": "CDC", "PS": "CDC", "TS": "CDC", "GH": "CDC",
    "FD": "FDA",
}
# NIH contract numbers start with the contracting office; these three map cleanly
NIH_CONTRACT_OFFICE = {"75N91": "NCI", "75N93": "NIAID", "75N92": "NHLBI"}
NSF_TOPIC_NAMES = {"BM": "Biomedical technologies", "MD": "Medical devices", "DH": "Digital health",
                   "PT": "Pharmaceutical technologies", "BT": "Biological technologies"}
# NSF topic codes that are biomedical by NSF's own definition. Used ONLY as an
# independent test set and as a rule-based inclusion; never for training.
NSF_BIOMED = {"BM": "Biomedical technologies", "MD": "Medical devices",
              "DH": "Digital health", "PT": "Pharmaceutical technologies"}
NSF_AMBIGUOUS = {"BT"}  # Biological technologies: part biomedical, part ag/industrial

NAME_PREFIX = {"dr", "mr", "mrs", "ms", "prof", "professor"}
NAME_SUFFIX = {"phd", "md", "dvm", "mba", "ms", "msc", "pe", "jr", "sr", "ii", "iii", "dds",
               "pharmd", "scd", "mph", "rn", "esq", "meng", "bs", "ma"}
CO_SUFFIX = re.compile(
    r"\b(incorporated|inc|llc|l l c|corp|corporation|co|company|ltd|limited|lp|llp|pllc|pbc|the)\b", re.I)
BOILERPLATE = re.compile(
    r"(project summary|project narrative|abstract|public health relevance( statement)?|"
    r"the broader impact\s*/?\s*(and\s*)?commercial potential of this|"
    r"small business (innovation research|technology transfer)|"
    r"\b(sbir|sttr)\b|phase (i{1,3}|1|2)\b|this award reflects nsf'?s statutory mission.*$)",
    re.I)


def clean_ws(s):
    s = s.fillna("").astype(str).str.replace(r"&(nbsp|amp|quot|#\d+);?|\bnbsp\b", " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def person_key(name, full=False):
    """'Trantolo, Debra J' / 'Dr. Debra J. Trantolo, PhD' -> 'debra trantolo'."""
    n = str(name or "").lower().replace(".", "")
    if "," in n:                                  # 'Last, First M'  or  'First Last, PhD'
        head, _, tail = n.partition(",")
        tail_words = re.sub(r"[^a-z ]", " ", tail).split()
        if tail_words and not all(w in NAME_SUFFIX for w in tail_words):
            n = f"{tail} {head}"
        else:
            n = head
    parts = re.sub(r"[^a-z\-' ]", " ", n).split()
    while parts and parts[0] in NAME_PREFIX:
        parts.pop(0)
    while len(parts) > 2 and parts[-1] in NAME_SUFFIX:
        parts.pop()
    if full or len(parts) < 2:
        return " ".join(parts)
    first, last = parts[0], parts[-1]
    if len(first) == 1 and len(parts) > 2:        # 'J Robert Smith' -> 'robert smith'
        first = parts[1]
    return f"{first} {last}"


def person_display(key):
    return " ".join(w.capitalize() if not re.search(r"[-']", w) else
                    re.sub(r"(^|[-'])([a-z])", lambda m: m.group(1) + m.group(2).upper(), w)
                    for w in key.split())


def org_key(name):
    n = re.sub(r"[^A-Za-z0-9& ]", " ", str(name or "")).lower()
    n = CO_SUFFIX.sub(" ", n)
    return re.sub(r"\s+", " ", n).strip()


def ri_key(name):
    n = org_key(name)
    n = re.sub(r"^(regents of|board of trustees of|trustees of|president and fellows of|rector and visitors of)\s+", "", n)
    n = re.sub(r"\b(office of sponsored (programs|research)|sponsored programs.*)$", "", n).strip()
    return n


def nice_org(name):
    n = re.sub(r"\s+", " ", str(name)).strip()
    if n.isupper() or n.islower():
        n = n.title()
        n = re.sub(r"\b(Llc|Inc|Usa|Us|Ii|Iii|Lp|Llp|Pbc|Rna|Dna|Ai|Md|3D)\b", lambda m: m.group(0).upper(), n)
        n = re.sub(r"\b(Of|And|For|The|At|In)\b", lambda m: m.group(0).lower(), n)
        n = n[0].upper() + n[1:] if n else n
        n = n.replace("'S", "'s")
    return n


def load_and_clean(csv_path):
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    n_raw = len(df)
    df.columns = [c.strip() for c in df.columns]
    for c in ["Company", "Award Title", "Agency", "Branch", "Phase", "Program", "Topic Code",
              "Agency Tracking Number", "Contract", "City", "State", "Zip", "Abstract",
              "PI Name", "PI Email", "RI Name"]:
        df[c] = clean_ws(df[c])
    df = df[df["Agency"] != ""]
    df.loc[df["Company"] == "", "Company"] = "(company not stated)"
    df["year"] = pd.to_numeric(df["Award Year"], errors="coerce").astype("Int64")
    df["amount"] = pd.to_numeric(df["Award Amount"].str.replace(r"[$,]", "", regex=True), errors="coerce")
    df["award_date"] = pd.to_datetime(df["Proposal Award Date"], errors="coerce")
    df["duns"] = df["Duns"].fillna("").str.replace(r"\D", "", regex=True)
    df["zip5"] = df["Zip"].str.extract(r"(\d{5})")[0].fillna("")
    df["State"] = df["State"].str.upper()
    df["City"] = df["City"].str.title()
    df["agency"] = df["Agency"].map(AGENCY_SHORT).fillna(df["Agency"])
    df["phase"] = df["Phase"].str.replace("Phase ", "", regex=False)
    for flag in ["Women Owned", "HUBZone Owned", "Socially and Economically Disadvantaged"]:
        df[flag] = df[flag].fillna("N").str.upper().str.startswith("Y")
    df = df.dropna(subset=["year", "amount"])
    df = df[df["amount"] > 0]
    # exact re-listings of the same award
    before = len(df)
    df = df.drop_duplicates(subset=["Agency", "Agency Tracking Number", "Contract", "Phase",
                                    "Award Title", "Award Amount", "Company"])
    stats = {"rows_raw": n_raw, "rows_after_cleaning": len(df), "duplicates_removed": before - len(df)}
    return df.reset_index(drop=True), stats


def add_entities(df):
    df["company_key"] = df["Company"].map(org_key)
    empty = df["company_key"] == ""                 # e.g. a firm literally called "THE CO."
    df.loc[empty, "company_key"] = df.loc[empty, "Company"].str.lower()
    df["company"] = df.groupby("company_key")["Company"].transform(lambda s: nice_org(s.mode().iat[0]))
    df["pi_key"] = df["PI Name"].map(person_key)
    full = df["PI Name"].map(lambda n: person_display(person_key(n, full=True)))
    df["pi"] = full.groupby(df["pi_key"]).transform(lambda s: max(s.mode(), key=len))
    df.loc[df["pi_key"] == "", "pi"] = ""
    df["ri_key"] = df["RI Name"].map(ri_key)
    df.loc[df["ri_key"].isin({"", "stub", "none", "n a", "na", "tbd"}), "ri_key"] = ""
    df["ri"] = ""
    m = df["ri_key"] != ""
    df.loc[m, "ri"] = df[m].groupby("ri_key")["RI Name"].transform(lambda s: nice_org(s.mode().iat[0]))
    return df


def add_program_detail(df):
    """'Program within the agency': NIH institute, DoD component, NSF topic area, etc."""
    sub = df["Branch"].copy()
    hhs = df["agency"] == "HHS"
    atn = df["Agency Tracking Number"].str.upper()
    ic = atn.str.extract(r"^\d?[A-Z][A-Z0-9]{2}([A-Z]{2})\d", expand=False)       # R43NS132653, SB1DK1...
    activity = atn.str.extract(r"^\d?([A-Z][A-Z0-9]{2})[A-Z]{2}\d", expand=False)
    inst = ic.map(NIH_IC)
    inst = inst.fillna(atn.str[:5].map(NIH_CONTRACT_OFFICE))                       # 75N91021C00023
    tc = df["Topic Code"].str.upper()
    inst = inst.fillna(tc.where(tc.isin(set(NIH_IC.values()) | {"NIDILRR", "NIOSH", "CDC", "FDA"})))
    inst = inst.where(hhs)
    inst = inst.fillna(df["Branch"].map({
        "Centers for Disease Control and Prevention": "CDC",
        "Food and Drug Administration": "FDA",
        "Administration for Community Living": "ACL / NIDILRR",
        "National Institutes of Health": "NIH (institute not stated)"}).where(hhs))
    inst = inst.replace({"NIDILRR": "ACL / NIDILRR", "NIOSH": "CDC"})
    sub = sub.where(~hhs, inst.fillna("HHS (other)"))
    nsf = df["agency"] == "NSF"
    sub = sub.where(~nsf, "NSF: " + tc.map(NSF_TOPIC_NAMES).fillna("topic " + tc.where(tc != "", "?")))
    sub = sub.where(sub.fillna("") != "", df["agency"])
    df["subprogram"] = sub
    df["nih_institute"] = inst.where(hhs & inst.fillna("").str.match(r"^N[A-Z ]+$|^NIH OD$"))
    df["activity_code"] = activity.where(hhs)
    return df


def add_geo(df):
    try:
        import zipcodes
    except ImportError:
        print("! zipcodes not installed; map coordinates skipped (pip install zipcodes)")
        df["lat"] = np.nan
        df["lon"] = np.nan
        return df
    cache = {}

    def look(z):
        if z not in cache:
            try:
                hit = zipcodes.matching(z) if z else []
            except Exception:
                hit = []
            cache[z] = (float(hit[0]["lat"]), float(hit[0]["long"])) if hit else (np.nan, np.nan)
        return cache[z]
    ll = df["zip5"].map(look)
    df["lat"] = ll.map(lambda t: t[0])
    df["lon"] = ll.map(lambda t: t[1])
    # fall back to the city's median coordinates when a ZIP is unknown
    city_med = df.dropna(subset=["lat"]).groupby(["City", "State"])[["lat", "lon"]].median()
    miss = df["lat"].isna()
    fb = df.loc[miss, ["City", "State"]].merge(city_med, left_on=["City", "State"], right_index=True, how="left")
    df.loc[miss, "lat"] = fb["lat"].values
    df.loc[miss, "lon"] = fb["lon"].values
    return df


def make_text(df):
    t = (df["Award Title"] + ". " + df["Abstract"]).str.replace(BOILERPLATE, " ", regex=True)
    return t.str.replace(r"\s+", " ", regex=True).str.strip()


# --------------------------------------------------------------------------
# 2. Embeddings
# --------------------------------------------------------------------------
EXTRA_STOP = ["project", "proposed", "propose", "proposal", "develop", "development", "developed",
              "technology", "technologies", "use", "used", "using", "based", "new", "novel", "system",
              "systems", "research", "aim", "aims", "specific", "will", "also", "provide", "including",
              "phase", "sbir", "sttr", "innovation", "innovative", "small", "business", "commercial",
              "potential", "impact", "broader", "award", "nsf", "nih", "navy", "army", "air", "force",
              "nasa", "doe", "dod", "high", "low", "approach", "current", "currently", "need", "needs",
              "design", "work", "effort", "program", "application", "applications", "solution", "study",
              "studies", "results", "product", "products", "market", "company", "llc", "inc", "tm"]


def embed_tfidf(texts):
    """Baseline: TF-IDF for classification, 200-d LSA for clustering."""
    from sklearn.feature_extraction import text as sktext
    stop = list(sktext.ENGLISH_STOP_WORDS.union(EXTRA_STOP))
    vec = TfidfVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=5, max_df=0.4,
                          sublinear_tf=True, max_features=120_000,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b")
    X = vec.fit_transform(texts)
    svd = TruncatedSVD(n_components=200, random_state=SEED)
    dense = normalize(svd.fit_transform(X))
    return X, dense


def embed_specter2(titles, abstracts, batch_size=32, cache=None):
    """SPECTER2 (allenai/specter2_base + proximity adapter): title [SEP] abstract -> CLS vector."""
    if cache and Path(cache).exists():
        return np.load(cache)
    import torch
    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel
    tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(dev).eval()
    docs = [t + tok.sep_token + a for t, a in zip(titles, abstracts)]
    order = np.argsort([len(d) for d in docs])          # batch similar lengths together
    out = np.zeros((len(docs), 768), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(docs), batch_size):
            idx = order[i:i + batch_size]
            enc = tok([docs[j] for j in idx], padding=True, truncation=True, max_length=512,
                      return_tensors="pt", return_token_type_ids=False).to(dev)
            out[idx] = model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy()
            if (i // batch_size) % 50 == 0:
                print(f"  specter2: {i}/{len(docs)}")
    out = normalize(out)
    if cache:
        np.save(cache, out)
    return out


# --------------------------------------------------------------------------
# 3. Biomedical vs. everything else
# --------------------------------------------------------------------------
LEXICON = re.compile(
    r"\b(patient|patients|clinical|clinician|disease|diseases|cancer|tumou?r|therap\w+|drug|drugs|"
    r"vaccine\w*|diagnos\w+|medical|medicine|surg\w+|biomedical|pharma\w+|infection\w*|pathogen\w*|"
    r"antibod\w+|hospital\w*|wound\w*|trauma\w*|injur\w+|rehabilit\w+|prosthe\w+|neuro\w+|cardi\w+|"
    r"diabet\w+|alzheimer\w*|genom\w+|protein\w*|cell therapy|in vivo|in vitro|fda|health\w*)\b", re.I)


def classify_biomedical(df, X_clf, C=4.0):
    """
    Positive-unlabelled set-up with 5-fold cross-fitting: every award is scored
    by a model that never saw that award.

    Labels the agencies themselves supply:
      HHS awards (NIH, CDC, FDA, ACL)                       -> biomedical
      DoD Defense Health Program awards                     -> biomedical
      NSF topic codes BM / MD / DH / PT                     -> biomedical
      NSF awards under any other topic code (except BT)     -> test negatives

    Two models are fitted.
      transfer : positives = HHS only. NSF and DHP awards are then a true
                 out-of-agency test: can text learnt from NIH find biomedical
                 work written in another agency's style?
      final    : positives = HHS + DHP + NSF biomedical codes. Still scored
                 out-of-fold, so the NSF/DHP numbers remain honest, but they are
                 now an in-agency test.
    Awards from any other agency that the final model scores above the
    threshold are the biomedical work an agency filter would have missed.
    """
    tc = df["Topic Code"].str.upper()
    hhs = (df["agency"] == "HHS").values
    nsf = (df["agency"] == "NSF").values
    dhp = (df["Branch"] == "Defense Health Program").values
    val_pos = nsf & tc.isin(NSF_BIOMED).values
    val_neg = nsf & ~tc.isin(set(NSF_BIOMED) | NSF_AMBIGUOUS).values & (tc != "").values
    val = val_pos | val_neg
    yv = val_pos[val].astype(int)
    rng = np.random.RandomState(SEED)
    half = rng.rand(val.sum()) < 0.5
    grid = np.linspace(0.05, 0.95, 91)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    def prf(y, yhat):
        return {"precision": round(float(precision_score(y, yhat, zero_division=0)), 3),
                "recall": round(float(recall_score(y, yhat, zero_division=0)), 3),
                "f1": round(float(f1_score(y, yhat, zero_division=0)), 3), "n": int(len(y))}

    def fit(seed):
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        p = cross_val_predict(clf, X_clf, seed.astype(int), cv=cv, method="predict_proba")[:, 1]
        pv = p[val]
        # threshold picked on one half of the NSF test set, reported on the other half
        thr = float(grid[np.argmax([f1_score(yv[half], pv[half] >= t) for t in grid])])
        return p, thr, {
            "threshold": round(thr, 2),
            "nsf_auc": round(float(roc_auc_score(yv, pv)), 4),
            "nsf_held_out_half": prf(yv[~half], pv[~half] >= thr),
            "defense_health_program_recall": round(float((p[dhp] >= thr).mean()), 3) if dhp.any() else None,
            "hhs_recall": round(float((p[hhs] >= thr).mean()), 3),
        }

    _, _, transfer = fit(hhs)
    p, thr, final = fit(hhs | dhp | val_pos)
    df["p_biomed"] = p

    lex_hits = (df["Award Title"] + " " + df["Abstract"]).str.count(LEXICON).values
    report = {
        "nsf_test_size": {"positives_BM_MD_DH_PT": int(yv.sum()), "negatives_other_codes": int((1 - yv).sum())},
        "transfer_model_hhs_seeds_only": transfer,
        "final_model": final,
        "keyword_baseline_3plus_hits": {
            "nsf": prf(yv, lex_hits[val] >= 3),
            "defense_health_program_recall": round(float((lex_hits[dhp] >= 3).mean()), 3) if dhp.any() else None},
        "nsf_BT_share_flagged": round(float((p[nsf & (tc == "BT").values] >= thr).mean()), 3),
    }
    why = np.full(len(df), "", dtype=object)
    why[p >= thr] = "Text model"
    why[val_pos] = "NSF topic code"
    why[dhp] = "Defense Health Program"
    why[hhs] = "HHS award"
    df["biomed_basis"] = why
    df["is_biomed"] = why != ""
    report["biomedical_by_basis"] = df.loc[df["is_biomed"], "biomed_basis"].value_counts().to_dict()
    report["flagged_outside_hhs_by_agency"] = (
        df[df["is_biomed"] & (df["agency"] != "HHS")]["agency"].value_counts().to_dict())
    report["share_of_each_agency_flagged"] = df.groupby("agency")["is_biomed"].mean().round(3).to_dict()
    return df, report


# --------------------------------------------------------------------------
# 4. Topics (k-means on the embedding + class-based TF-IDF labels)
# --------------------------------------------------------------------------
def topic_model(texts, dense, k, label_file_key=None, labels_path=None):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    lab = km.fit_predict(dense)
    from sklearn.feature_extraction import text as sktext
    stop = list(sktext.ENGLISH_STOP_WORDS.union(EXTRA_STOP))
    cv = CountVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=5, max_df=0.5,
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b", max_features=60_000)
    counts = cv.fit_transform(texts)
    vocab = np.array(cv.get_feature_names_out())
    per = np.vstack([np.asarray(counts[lab == c].sum(axis=0)).ravel() for c in range(k)]).astype(float)
    tf = per / per.sum(axis=1, keepdims=True)
    idf = np.log(1 + per.sum(axis=1).mean() / (per.sum(axis=0) + 1))
    ctfidf = tf * idf
    terms = [list(vocab[np.argsort(-ctfidf[c])[:12]]) for c in range(k)]
    names = {}
    if labels_path and Path(labels_path).exists():
        names = json.loads(Path(labels_path).read_text()).get(label_file_key, {})
    topics = []
    for c in range(k):
        auto = ", ".join(terms[c][:3])
        topics.append({"id": c, "label": names.get(str(c), auto), "terms": terms[c],
                       "n": int((lab == c).sum())})
    return lab, topics


# --------------------------------------------------------------------------
# 5. Teams and export
# --------------------------------------------------------------------------
def team_table(bio):
    g = bio[bio["pi_key"] != ""].groupby(["company", "pi"])
    t = g.agg(awards=("amount", "size"), total=("amount", "sum"),
              programs=("Program", lambda s: "+".join(sorted(set(s)))),
              phases=("phase", lambda s: "+".join(sorted(set(s)))),
              agencies=("agency", lambda s: "+".join(sorted(set(s)))),
              subprograms=("subprogram", lambda s: "; ".join(sorted(set(s)))),
              partners=("ri", lambda s: "; ".join(sorted(set(x for x in s if x)))),
              first_year=("year", "min"), last_year=("year", "max")).reset_index()
    t["both_programs"] = t["programs"].str.contains(r"\+")
    t["both_phases"] = t["phases"].str.contains(r"\+")
    t["multi_agency"] = t["agencies"].str.contains(r"\+")
    return t.sort_values(["both_programs", "multi_agency", "both_phases", "total"], ascending=False)


def export_for_page(df, bio, topics, fields, report, meta):
    def index(series):
        vals = sorted(v for v in series.unique() if v)
        return vals, {v: i for i, v in enumerate(vals)}
    companies, ci = index(bio["company"])
    people, pi_ = index(bio["pi"])
    ris, ri_ = index(bio["ri"])
    subs, si = index(bio["subprogram"])
    agencies, ai = index(bio["agency"])
    bases, bi = index(bio["biomed_basis"])

    loc = bio.groupby("company").agg(city=("City", "first"), state=("State", "first"),
                                     lat=("lat", "median"), lon=("lon", "median"),
                                     w=("Women Owned", "max"), h=("HUBZone Owned", "max"),
                                     d=("Socially and Economically Disadvantaged", "max"))
    comp_rows = []
    for c in companies:
        r = loc.loc[c]
        comp_rows.append([c, r.city, r.state,
                          None if pd.isna(r.lat) else round(float(r.lat), 3),
                          None if pd.isna(r.lon) else round(float(r.lon), 3),
                          int(r.w) + 2 * int(r.h) + 4 * int(r.d)])
    awards = []
    for r in bio.rename(columns={'Award Title': 'title'}).itertuples():
        awards.append([int(r.year), int(round(r.amount)), ai[r.agency], si[r.subprogram],
                       0 if r.Program == "SBIR" else 1, 1 if r.phase == "I" else 2,
                       int(r.topic), ci[r.company], pi_.get(r.pi, -1), ri_.get(r.ri, -1),
                       bi[r.biomed_basis], r.title[:160]])
    return {
        "meta": meta, "quality": report,
        "agencies": agencies, "subprograms": subs, "bases": bases,
        "companies": comp_rows, "people": people, "ris": ris,
        "topics": topics, "fields": fields,
        "awardCols": ["year", "amount", "agency", "sub", "program", "phase", "topic",
                      "company", "pi", "ri", "basis", "title"],
        "awards": awards,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="build")
    ap.add_argument("--embedder", choices=["tfidf", "specter2"], default="tfidf")
    ap.add_argument("--years", type=int, default=5, help="number of award years to keep")
    ap.add_argument("--end-year", type=int, default=None, help="last award year (default: latest in file)")
    ap.add_argument("--k-biomed", type=int, default=16, help="number of biomedical topics")
    ap.add_argument("--k-fields", type=int, default=14, help="number of non-biomedical fields")
    ap.add_argument("--labels", default="topic_labels.json", help="optional hand-written topic names")
    ap.add_argument("--template", default="dashboard_template.html")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print("1/6 cleaning")
    df, stats = load_and_clean(a.csv)
    end = a.end_year or int(df["year"].max())
    start = end - a.years + 1
    df = df[(df["year"] >= start) & (df["year"] <= end)].reset_index(drop=True)
    stats.update(window=[start, end], rows_in_window=len(df),
                 awards_per_year={int(k): int(v) for k, v in df["year"].value_counts().sort_index().items()})
    df = add_geo(add_program_detail(add_entities(df)))
    text = make_text(df)

    print(f"2/6 embedding {len(df):,} awards with {a.embedder}")
    if a.embedder == "specter2":
        dense = embed_specter2(df["Award Title"].tolist(),
                               df["Abstract"].str.replace(BOILERPLATE, " ", regex=True).tolist(),
                               cache=out / "specter2_embeddings.npy")
        X_clf = dense
    else:
        X_clf, dense = embed_tfidf(text)

    print("3/6 separating biomedical awards")
    df, report = classify_biomedical(df, X_clf, C=4.0 if a.embedder == "tfidf" else 1.0)
    report["embedder"] = a.embedder

    print("4/6 topics")
    b = df["is_biomed"].values
    lab_b, topics = topic_model(text[b], dense[b], a.k_biomed, "biomedical", a.labels)
    lab_o, fields = topic_model(text[~b], dense[~b], a.k_fields, "fields", a.labels)
    df["topic"] = -1
    df.loc[b, "topic"] = lab_b
    df["field"] = "Biomedical"
    df.loc[~b, "field"] = [fields[i]["label"] for i in lab_o]
    for t in topics:
        m = df["topic"] == t["id"]
        t["amount"] = float(df.loc[m, "amount"].sum())
    for i, f in enumerate(fields):
        m = (~b) & (df["field"] == f["label"])
        sel = df[~b][lab_o == i]
        f["amount"] = float(sel["amount"].sum())
        f["agencies"] = sel["agency"].value_counts().head(3).to_dict()
    fields.append({"id": -1, "label": "Biomedical", "terms": [], "n": int(b.sum()),
                   "amount": float(df.loc[b, "amount"].sum()),
                   "agencies": df.loc[b, "agency"].value_counts().head(3).to_dict()})
    # does the unsupervised topic structure line up with NIH's own institutes?
    nih = df[b & df["nih_institute"].notna().values]
    report["topics_vs_nih_institute_nmi"] = round(float(
        normalized_mutual_info_score(nih["nih_institute"], nih["topic"])), 3)

    print("5/6 teams")
    bio = df[df["is_biomed"]].copy()
    teams = team_table(bio)
    report["teams"] = {"n": int(len(teams)), "both_programs": int(teams.both_programs.sum()),
                       "both_phases": int(teams.both_phases.sum()),
                       "multi_agency": int(teams.multi_agency.sum())}
    geo_missing = int(bio["lat"].isna().sum())

    print("6/6 writing")
    keep = ["company", "Award Title", "agency", "Branch", "subprogram", "nih_institute", "Program", "phase",
            "year", "amount", "Agency Tracking Number", "Contract", "Topic Code", "City", "State", "zip5",
            "lat", "lon", "pi", "PI Email", "ri", "Women Owned", "HUBZone Owned",
            "Socially and Economically Disadvantaged", "is_biomed", "biomed_basis", "p_biomed",
            "topic", "field", "Abstract"]
    df[keep].to_csv(out / "awards_clean_5yr.csv", index=False)
    bio[keep].to_csv(out / "biomedical_awards.csv", index=False)
    teams.to_csv(out / "teams.csv", index=False)
    (out / "topics.json").write_text(json.dumps({"biomedical": topics, "fields": fields}, indent=1))
    meta = {**stats, "embedder": a.embedder, "biomedical_awards": int(len(bio)),
            "biomedical_amount": float(bio["amount"].sum()), "all_amount": float(df["amount"].sum()),
            "not_on_map": geo_missing}
    (out / "quality_report.json").write_text(json.dumps({**meta, **report}, indent=1))
    data = export_for_page(df, bio, topics, fields, report, meta)
    blob = json.dumps(data, separators=(",", ":"))
    (out / "data.json").write_text(blob)
    tpl = Path(a.template)
    if tpl.exists():
        (out / "index.html").write_text(tpl.read_text().replace("/*__DATA__*/null", blob.replace("</", "<\\/")))
        print(f"   page: {out/'index.html'} ({len(blob)/1e6:.1f} MB of data)")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
