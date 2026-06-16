#!/usr/bin/env python3
"""
Parallel gene-region extraction from RefSeq GenBank files.

RefSeq .genomic.gbff.gz files contain gene annotations but most records
use CONTIG references instead of inline sequences. The actual genomic
sequences are in matching .genomic.fna.gz files.

For each GBFF file N, we load all matching FNA N.* files to build a
sequence index (accession -> sequence), then extract gene regions using
annotations from the GBFF paired with sequences from the FNA index.
"""

import argparse
import gzip
import json
import logging
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Seq import UndefinedSequenceError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

GENE_TYPE_MAP = {
    "protein_coding": "<cds>",
    "mRNA": "<cds>",
    "pseudo": "<pseudo>",
    "pseudogene": "<pseudo>",
    "transcribed_pseudogene": "<pseudo>",
    "unprocessed_pseudogene": "<pseudo>",
    "processed_pseudogene": "<pseudo>",
    "tRNA": "<tRNA>",
    "rRNA": "<rRNA>",
    "ncRNA": "<ncRNA>",
    "lncRNA": "<ncRNA>",
    "lnc_RNA": "<ncRNA>",
    "snRNA": "<ncRNA>",
    "snoRNA": "<ncRNA>",
    "scRNA": "<ncRNA>",
    "miRNA": "<ncRNA>",
    "misc_RNA": "<misc_RNA>",
    "other": "<misc_RNA>",
}


def classify_gene(feature):
    for qual_key in ("gene_biotype", "biotype"):
        if qual_key in feature.qualifiers:
            biotype = feature.qualifiers[qual_key][0]
            if biotype in GENE_TYPE_MAP:
                return GENE_TYPE_MAP[biotype]
    if "pseudo" in feature.qualifiers or "pseudogene" in feature.qualifiers:
        return "<pseudo>"
    return "<cds>"


def load_fna_index(fna_files, needed_ids=None):
    """Load sequences from FNA files into a dict mapping accession -> uppercase sequence.
    
    If needed_ids is provided, only load sequences for those accessions to save memory.
    """
    seq_index = {}
    for fna_path in fna_files:
        opener = gzip.open if str(fna_path).endswith(".gz") else open
        mode = "rt" if str(fna_path).endswith(".gz") else "r"
        with opener(fna_path, mode) as fh:
            current_id = None
            current_lines = []
            for line in fh:
                if line.startswith(">"):
                    # Save previous record
                    if current_id is not None and (needed_ids is None or current_id in needed_ids):
                        seq_index[current_id] = "".join(current_lines).upper()
                    # Parse new header
                    current_id = line[1:].split()[0]
                    current_lines = []
                else:
                    if needed_ids is None or current_id in needed_ids:
                        current_lines.append(line.strip())
                    # else: skip accumulating sequence data for unneeded records
            # Last record
            if current_id is not None and (needed_ids is None or current_id in needed_ids):
                seq_index[current_id] = "".join(current_lines).upper()
    return seq_index


def scan_gbff_accessions(gbff_path):
    """Quick scan of a GBFF file to collect accession IDs of records that have gene features."""
    opener = gzip.open if str(gbff_path).endswith(".gz") else open
    mode = "rt" if str(gbff_path).endswith(".gz") else "r"
    ids_with_genes = set()
    with opener(gbff_path, mode) as handle:
        for record in SeqIO.parse(handle, "genbank"):
            has_gene = any(f.type == "gene" for f in record.features)
            if has_gene:
                try:
                    _ = str(record.seq)
                    # Has inline sequence — no need to fetch from FNA
                except UndefinedSequenceError:
                    ids_with_genes.add(record.id)
    return ids_with_genes


def extract_single_gbff(gbff_path, fna_files, species_type, out_file):
    """Extract gene regions from one GBFF using its paired FNA files for sequences."""
    # Step 1: Scan GBFF to find which accessions we actually need from FNA
    needed_ids = scan_gbff_accessions(gbff_path) if fna_files else set()

    # Step 2: Load only needed sequences from FNA files
    seq_index = load_fna_index(fna_files, needed_ids) if fna_files else {}

    # Step 2: Parse GBFF annotations and extract gene regions
    opener = gzip.open if str(gbff_path).endswith(".gz") else open
    mode = "rt" if str(gbff_path).endswith(".gz") else "r"

    rows = []
    records_processed = 0
    records_with_seq = 0
    records_from_fna = 0
    records_skipped = 0
    gene_type_counts = Counter()
    total_bp = 0

    with opener(gbff_path, mode) as handle:
        for record in SeqIO.parse(handle, "genbank"):
            records_processed += 1

            # Try inline sequence first, fall back to FNA index
            full_seq = None
            try:
                full_seq = str(record.seq).upper()
                records_with_seq += 1
            except UndefinedSequenceError:
                full_seq = seq_index.get(record.id)
                if full_seq is not None:
                    records_from_fna += 1
                else:
                    records_skipped += 1
                    continue

            if len(full_seq) < 100:
                continue

            for feature in record.features:
                if feature.type != "gene":
                    continue
                try:
                    start = int(feature.location.start)
                    end = int(feature.location.end)
                except Exception:
                    continue

                gene_len = end - start
                if gene_len < 50 or gene_len > 10_000_000:
                    continue

                seq = full_seq[start:end]
                if not seq:
                    continue
                if seq.count("N") / len(seq) > 0.10:
                    continue

                strand = "<+>" if feature.location.strand != -1 else "<->"
                gene_type = classify_gene(feature)

                rows.append(
                    {
                        "record_id": record.id,
                        "species_type": species_type,
                        "gene_type": gene_type,
                        "strand": strand,
                        "sequence": seq,
                        "start": start,
                        "end": end,
                    }
                )
                gene_type_counts[gene_type] += 1
                total_bp += len(seq)

    if rows:
        pd.DataFrame(rows).to_parquet(out_file, index=False, engine="pyarrow")

    return {
        "file": str(gbff_path),
        "records_processed": records_processed,
        "records_with_inline_seq": records_with_seq,
        "records_from_fna": records_from_fna,
        "records_skipped": records_skipped,
        "genes_extracted": len(rows),
        "total_bp": total_bp,
        "gene_type_counts": dict(gene_type_counts),
        "out_file": str(out_file),
        "fna_sequences_loaded": len(seq_index),
    }


def find_matching_fna(gbff_path, input_dir):
    """Find FNA files matching a GBFF file.
    
    GBFF: category.N.genomic.gbff.gz
    FNA:  category.N.M.genomic.fna.gz  (M = 1,2,3,...)
    """
    gbff_name = Path(gbff_path).name
    # Extract the category and number: e.g. "protozoa.1" from "protozoa.1.genomic.gbff.gz"
    m = re.match(r"^(.+?)\.(\d+)\.genomic\.gbff\.gz$", gbff_name)
    if not m:
        return []
    category_prefix = m.group(1)
    gbff_num = m.group(2)
    # Match: category.N.*.genomic.fna.gz
    pattern = f"{category_prefix}.{gbff_num}.*.genomic.fna.gz"
    fna_files = sorted(Path(input_dir).glob(pattern))
    return [str(f) for f in fna_files]


def main():
    parser = argparse.ArgumentParser(description="Parallel extraction of gene regions")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--species_type", required=True)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    gbff_files = sorted(Path(args.input_dir).glob("*.genomic.gbff.gz"))
    if not gbff_files:
        gbff_files = sorted(Path(args.input_dir).glob("*.gbff.gz"))
    if not gbff_files:
        raise SystemExit(f"No GenBank files found in {args.input_dir}")

    logger.info("Found %d GBFF files in %s", len(gbff_files), args.input_dir)

    # Build the GBFF -> FNA mapping
    gbff_fna_pairs = []
    total_fna = 0
    for gbff in gbff_files:
        fna_files = find_matching_fna(str(gbff), args.input_dir)
        total_fna += len(fna_files)
        gbff_fna_pairs.append((str(gbff), fna_files))
        if not fna_files:
            logger.warning("No FNA files found for %s — will use inline sequences only", gbff.name)

    logger.info("Total FNA files matched: %d", total_fna)
    logger.info("Using %d workers", args.workers)

    total_genes = 0
    total_bp = 0
    total_records = 0
    total_fna_used = 0
    total_skipped = 0
    gene_type_counts = Counter()
    num_parquet = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for i, (gbff, fna_files) in enumerate(gbff_fna_pairs):
            out_file = Path(args.output_dir) / f"part_{i:06d}.parquet"
            fut = ex.submit(extract_single_gbff, gbff, fna_files, args.species_type, str(out_file))
            futures[fut] = gbff

        failed_files = []
        for idx, fut in enumerate(as_completed(futures), start=1):
            gbff_name = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                logger.error("FAILED %s: %s", gbff_name, exc)
                failed_files.append(gbff_name)
                continue
            total_genes += result["genes_extracted"]
            total_bp += result["total_bp"]
            total_records += result["records_processed"]
            total_fna_used += result["records_from_fna"]
            total_skipped += result["records_skipped"]
            gene_type_counts.update(result["gene_type_counts"])
            if result["genes_extracted"] > 0:
                num_parquet += 1
            if idx % 20 == 0 or idx == len(futures):
                logger.info(
                    "Completed %d/%d files; genes=%s bp=%.2fB (fna_resolved=%d skipped=%d)",
                    idx, len(futures), f"{total_genes:,}", total_bp / 1e9,
                    total_fna_used, total_skipped,
                )

    if failed_files:
        logger.warning("Failed to extract %d files: %s", len(failed_files), failed_files)

    stats = {
        "species_type": args.species_type,
        "total_genes": total_genes,
        "total_bp": total_bp,
        "total_bp_billions": round(total_bp / 1e9, 2),
        "mean_gene_length": round(total_bp / max(total_genes, 1), 1),
        "gene_type_counts": dict(gene_type_counts),
        "num_gbff_files": len(gbff_files),
        "num_fna_files": total_fna,
        "records_processed": total_records,
        "records_resolved_from_fna": total_fna_used,
        "records_skipped_no_seq": total_skipped,
        "num_parquet_batches": num_parquet,
        "workers": args.workers,
    }

    with open(Path(args.output_dir) / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Extraction complete: genes=%s bp=%.2fB output=%s", f"{total_genes:,}", total_bp / 1e9, args.output_dir)


if __name__ == "__main__":
    main()
