#!/usr/bin/env python3
"""
FT-1 Step 2: Extract Evo2 7B embeddings for all BRCA1 variant sequences.

Produces:
    data/brca1_ref_embeddings.npy   (n_variants, hidden_dim)
    data/brca1_alt_embeddings.npy   (n_variants, hidden_dim)
    data/brca1_features_delta.npy   (alt - ref)
    data/brca1_features_concat.npy  (ref ; alt ; delta)

Usage:
    python extract_embeddings.py [--layer 20] [--gpu 0]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from utils.embeddings import extract_embeddings_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(
        os.path.dirname(__file__), "data"))
    parser.add_argument("--layer", type=int, default=config.EMBEDDING_LAYER)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    ref_seqs = np.load(os.path.join(args.data_dir, "brca1_ref_seqs.npy"),
                       allow_pickle=True)
    alt_seqs = np.load(os.path.join(args.data_dir, "brca1_alt_seqs.npy"),
                       allow_pickle=True)
    print(f"Loaded {len(ref_seqs)} ref + {len(alt_seqs)} alt sequences")

    # Load model
    from evo2 import Evo2  # heavy import — keep inside main
    model = Evo2("evo2_7b")

    print("Extracting reference embeddings …")
    ref_embs = extract_embeddings_batch(model, ref_seqs, args.layer)

    print("Extracting alternate embeddings …")
    alt_embs = extract_embeddings_batch(model, alt_seqs, args.layer)

    delta = alt_embs - ref_embs
    concat = np.concatenate([ref_embs, alt_embs, delta], axis=1)

    np.save(os.path.join(args.data_dir, "brca1_ref_embeddings.npy"), ref_embs)
    np.save(os.path.join(args.data_dir, "brca1_alt_embeddings.npy"), alt_embs)
    np.save(os.path.join(args.data_dir, "brca1_features_delta.npy"), delta)
    np.save(os.path.join(args.data_dir, "brca1_features_concat.npy"), concat)
    print(f"Saved embeddings → {args.data_dir}  (hidden_dim={ref_embs.shape[1]})")


if __name__ == "__main__":
    main()
