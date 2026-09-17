"""
End-to-end preprocessing for:
  M1 = SVM + TF-IDF
  M2 = Logistic Regression + TF-IDF
  M3 = AfriBERTa-Base
  M4 = XLM-R-Base
  M5 = Fine-tuned Best (XLM-R)
Across three splits: 70-30, 85-15, 80-20
"""
import os, json, random, argparse
import numpy as np
import pandas as pd
import yaml
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

from preprocess import clean_corpus
from tfidf_features import build_tfidf, fit_transform_pair, transform_pair
from tokenizer_utils import load_tokenizer, encode_texts
from augment import augment_text

MODEL_CONFIGS = {
    "M1_SVM_TFIDF":       {"type": "tfidf",     "tokenizer": None},
    "M2_LogReg_TFIDF":    {"type": "tfidf",     "tokenizer": None},
    "M3_AfriBERTa_Base":  {"type": "transformer","tokenizer": "castorini/afriberta_base"},
    "M4_XLMR_Base":       {"type": "transformer","tokenizer": "xlm-roberta-base"},
    "M5_FineTuned_Best":  {"type": "transformer","tokenizer": "xlm-roberta-base"},
}

SPLIT_LABELS = {0.30: "70-30", 0.15: "85-15", 0.20: "80-20"}

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def load_raw(cfg):
    df = pd.read_csv(cfg["data"]["raw_path"])
    df = df[[cfg["data"]["text_col"], cfg["data"]["label_col"]]].dropna()
    df.columns = ["text", "label"]
    return df

def make_splits(df, test_frac, val_frac_of_train, seed):
    X = df["text"].tolist(); y = df["label"].tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_frac, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=val_frac_of_train,
        random_state=seed, stratify=y_tr)
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)

def clean_splits(splits, cfg):
    kw = dict(
        strip_html_flag=cfg["clean"]["strip_html"],
        remove_urls_flag=cfg["clean"]["remove_urls"],
        remove_mentions_flag=cfg["clean"]["remove_mentions"],
        normalize_unicode_flag=cfg["clean"]["normalize_unicode"],
        collapse_whitespace_flag=cfg["clean"]["collapse_whitespace"],
        lowercase=cfg["clean"]["lowercase"],
    )
    return [(clean_corpus(X, **kw), y) for X, y in splits]

def augment_train(X, y, cfg):
    if not cfg["augment"]["enabled"]:
        return X, y
    out_X, out_y = [], []
    for x, lab in zip(X, y):
        out_X.append(x); out_y.append(lab)
        out_X.append(augment_text(
            x,
            synonym_replace_p=cfg["augment"]["synonym_replace_p"],
            random_swap_p=cfg["augment"]["random_swap_p"],
            random_deletion_p=cfg["augment"]["random_deletion_p"],
        ))
        out_y.append(lab)
    return out_X, out_y

# ---------- TF-IDF path (M1, M2) ----------
def process_tfidf(model_id, splits, cfg, out_dir, split_label):
    word_vec, char_vec = build_tfidf(cfg["tfidf"], cfg["tfidf"])
    X_tr, y_tr = splits[0]
    X_va, y_va = splits[1]
    X_te, y_te = splits[2]

    Xtr = fit_transform_pair(word_vec, char_vec, X_tr)
    Xva = transform_pair(word_vec, char_vec, X_va)
    Xte = transform_pair(word_vec, char_vec, X_te)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{model_id}_{split_label}")
    sp.save_npz(base + "_train.npz", Xtr)
    sp.save_npz(base + "_val.npz",   Xva)
    sp.save_npz(base + "_test.npz",  Xte)
    np.save(base + "_y_train.npy", np.array(y_tr))
    np.save(base + "_y_val.npy",   np.array(y_va))
    np.save(base + "_y_test.npy",  np.array(y_te))
    print(f"[OK] {model_id} [{split_label}] TF-IDF written.")

# ---------- Transformer path (M3, M4, M5) ----------
def process_transformer(model_id, splits, cfg, out_dir, split_label, tok_name):
    tokenizer = load_tokenizer(tok_name)
    max_len = cfg["tokenizer"]["max_len"]
    artifacts = {"model_id": model_id, "split": split_label,
                 "tokenizer": tok_name, "max_len": max_len}
    for name, (X, y) in zip(["train","val","test"], splits):
        enc = encode_texts(tokenizer, X, max_len=max_len)
        artifacts[f"{name}_input_ids"]      = enc["input_ids"]
        artifacts[f"{name}_attention_mask"] = enc["attention_mask"]
        artifacts[f"{name}_labels"]         = y
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_id}_{split_label}.json")
    with open(out_path, "w") as f:
        json.dump(artifacts, f)
    print(f"[OK] {model_id} [{split_label}] transformer encodings written.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])
    df = load_raw(cfg)

    for test_frac in cfg["data"]["splits"]:
        split_label = SPLIT_LABELS[test_frac]
        splits = make_splits(df, test_frac,
                             cfg["data"]["val_frac_of_train"],
                             cfg["data"]["seed"])
        splits = clean_splits(splits, cfg)

        # augment only training split
        X_tr, y_tr = augment_train(splits[0][0], splits[0][1], cfg)
        splits = [(X_tr, y_tr), splits[1], splits[2]]

        for model_id, m_cfg in MODEL_CONFIGS.items():
            if m_cfg["type"] == "tfidf":
                process_tfidf(model_id, splits, cfg,
                              cfg["data"]["out_dir"], split_label)
            else:
                process_transformer(model_id, splits, cfg,
                                    cfg["data"]["out_dir"],
                                    split_label, m_cfg["tokenizer"])

if __name__ == "__main__":
    main()
