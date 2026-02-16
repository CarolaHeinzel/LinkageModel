from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Sequence, Union, Tuple
import itertools
import math
import heapq

import numpy as np
import msprime


# This code simulates a island model with msprime
# Then, it chooses M markers based on Fst and calculates the allele frequencies of 
# them per population and the genetic distances between them in cM.
#%%


# ---------------------------
# Helper: choose migration rate to increase differentiation
# ---------------------------

def pairwise_m_for_target_fst(target_fst: float, Ne: float, K: int) -> float:
    """
    Rough island-model approximation:
        FST ≈ 1 / (1 + 4 * Ne * (K-1) * m_pairwise)
    Solve for m_pairwise:
        m ≈ (1/FST - 1) / (4 * Ne * (K-1))
    """
    if not (0 < target_fst < 1):
        raise ValueError("target_fst must be between 0 and 1.")
    if Ne <= 0:
        raise ValueError("Ne must be > 0.")
    if K < 2:
        return 0.0
    return (1.0 / target_fst - 1.0) / (4.0 * Ne * (K - 1))


# ---------------------------
# 1) Island model: simulate C chromosomes
# ---------------------------

ChromosomeSet = List[msprime.TreeSequence]
TreeSeqOrIter = Union[
    msprime.TreeSequence,                       # C==1, num_replicates is None
    ChromosomeSet,                              # C>1,  num_replicates is None
    Generator[ChromosomeSet, None, None],       # num_replicates provided
]


def simulate_island_model(
    K: int,
    n_diploids_per_island: int,
    *,
    C: int = 1,
    Ne: float = 10_000,
    m: float = 1e-3,                            # pairwise migration rate
    sequence_length: float = 1e6,
    recombination_rate: float = 1e-8,
    mutation_rate: float = 1e-8,
    dtwf_generations: Optional[int] = None,
    num_replicates: Optional[int] = None,
    random_seed: Optional[int] = 42,
) -> TreeSeqOrIter:
    """
    Simulate a symmetric K-deme island model and return C independent chromosomes.
    Diploid sampling is used via ploidy=2.
    """
    if K < 1:
        raise ValueError("K must be >= 1.")
    if n_diploids_per_island < 0:
        raise ValueError("n_diploids_per_island must be >= 0.")
    if C < 1:
        raise ValueError("C must be >= 1.")
    if Ne <= 0:
        raise ValueError("Ne must be > 0.")
    if m < 0:
        raise ValueError("m must be >= 0.")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be > 0.")
    if recombination_rate < 0:
        raise ValueError("recombination_rate must be >= 0.")
    if mutation_rate < 0:
        raise ValueError("mutation_rate must be >= 0.")
    if dtwf_generations is not None and dtwf_generations < 0:
        raise ValueError("dtwf_generations must be >= 0 if provided.")
    if num_replicates is not None and num_replicates < 1:
        raise ValueError("num_replicates must be >= 1 if provided.")

    demography = msprime.Demography.island_model([Ne] * K, migration_rate=m)
    samples: Dict[int, int] = {pop_id: n_diploids_per_island for pop_id in range(K)}

    if dtwf_generations is None:
        ancestry_model = "hudson"
    else:
        ancestry_model = [
            msprime.DiscreteTimeWrightFisher(duration=dtwf_generations),
            msprime.StandardCoalescent(),
        ]

    total = C if num_replicates is None else num_replicates * C

    ts_iter = msprime.sim_ancestry(
        samples=samples,
        demography=demography,
        ploidy=2,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        model=ancestry_model,
        num_replicates=total,
        random_seed=random_seed,
    )

    base_mut_seed = None if random_seed is None else random_seed + 1

    def mutate_stream() -> Generator[msprime.TreeSequence, None, None]:
        for i, ts in enumerate(ts_iter):
            mut_seed = None if base_mut_seed is None else base_mut_seed + i
            yield msprime.sim_mutations(ts, rate=mutation_rate, random_seed=mut_seed)

    muts = mutate_stream()

    if num_replicates is None:
        chroms = list(itertools.islice(muts, C))
        return chroms[0] if C == 1 else chroms

    def _generator() -> Generator[ChromosomeSet, None, None]:
        for _ in range(num_replicates):
            yield list(itertools.islice(muts, C))

    return _generator()


# ---------------------------
# 2) Select top M markers and compute inter-marker distances
# ---------------------------

TreeSeqOrSeq = Union[msprime.TreeSequence, Sequence[msprime.TreeSequence]]


@dataclass(frozen=True)
class SNPRecord:
    chrom: int
    pos_bp: float
    ref: str
    alt: str
    # Distance to the previous selected marker *on the same chromosome*
    delta_bp: Optional[float]
    delta_cM: Optional[float]
    alt_freq_by_pop: Dict[int, float]
    score_af_range: float                       # max(AF) - min(AF)


def top_markers_by_af_range(
    chroms: TreeSeqOrSeq,
    M: int,
    *,
    recombination_rate: float,
    strict_single_mutation: bool = True,
) -> List[SNPRecord]:
    """
    Pick the top-M biallelic, polymorphic SNPs with the largest allele-frequency
    range across populations and compute distances between consecutive selected
    markers on each chromosome.

    Inter-marker distances:
        delta_bp = current_pos - previous_selected_pos_on_same_chrom
        delta_cM = 100 * recombination_rate * delta_bp  (constant-r approximation)
    """
    if M < 1:
        raise ValueError("M must be >= 1")
    if recombination_rate < 0:
        raise ValueError("recombination_rate must be >= 0")

    chrom_list = [chroms] if isinstance(chroms, msprime.TreeSequence) else list(chroms)

    # Keep only the best M using a min-heap
    heap: List[Tuple[float, int, dict]] = []
    tie_id = 0

    for chrom_idx, ts in enumerate(chrom_list):
        samples = ts.samples()
        node_pops = ts.tables.nodes.population[samples]  # aligns with var.genotypes
        pop_ids = sorted({int(p) for p in node_pops})

        pop_masks = {pid: (node_pops == pid) for pid in pop_ids}
        pop_denoms = {pid: int(mask.sum()) for pid, mask in pop_masks.items()}

        for var in ts.variants():
            # Enforce biallelic sites
            if len(var.alleles) != 2:
                continue

            # Optionally enforce exactly one mutation at the site (avoid multi-hit)
            if strict_single_mutation and len(var.site.mutations) != 1:
                continue

            # SNP-like alleles
            if any(a is None or len(a) != 1 for a in var.alleles):
                continue

            g = var.genotypes

            # Enforce genotype states only in {0,1}
            if np.any(g < 0) or np.any(g > 1):
                continue

            # Enforce polymorphism
            alt_count = int(g.sum())
            n = len(g)
            if alt_count == 0 or alt_count == n:
                continue

            # Compute ALT allele frequency per population
            alt_freq_by_pop: Dict[int, float] = {}
            afs: List[float] = []
            for pid in pop_ids:
                denom = pop_denoms[pid]
                if denom == 0:
                    af = math.nan
                else:
                    af = float(g[pop_masks[pid]].sum()) / float(denom)
                alt_freq_by_pop[pid] = af
                afs.append(af)

            afs_clean = [x for x in afs if not math.isnan(x)]
            if len(afs_clean) == 0:
                continue

            # Selection score: allele-frequency range across populations
            score = max(afs_clean) - min(afs_clean)

            item = dict(
                chrom=chrom_idx,
                pos_bp=float(var.site.position),
                ref=var.alleles[0],
                alt=var.alleles[1],
                alt_freq_by_pop=alt_freq_by_pop,
                score=score,
            )

            tie_id += 1
            if len(heap) < M:
                heapq.heappush(heap, (score, tie_id, item))
            else:
                if score > heap[0][0]:
                    heapq.heapreplace(heap, (score, tie_id, item))

    if len(heap) == 0:
        return []

    # Extract the selected SNPs and sort by (chromosome, position)
    selected = [t[2] for t in heap]
    selected.sort(key=lambda x: (x["chrom"], x["pos_bp"]))

    # --- THIS BLOCK COMPUTES THE DISTANCES BETWEEN SELECTED MARKERS ---
    prev_pos_by_chrom: Dict[int, float] = {}
    out: List[SNPRecord] = []

    for it in selected:
        chrom = it["chrom"]
        pos = it["pos_bp"]

        if chrom in prev_pos_by_chrom:
            delta_bp = pos - prev_pos_by_chrom[chrom]
            delta_cM = 100.0 * recombination_rate * delta_bp
        else:
            # First selected marker on this chromosome has no previous marker
            delta_bp = None
            delta_cM = None

        prev_pos_by_chrom[chrom] = pos

        out.append(
            SNPRecord(
                chrom=chrom,
                pos_bp=pos,
                ref=it["ref"],
                alt=it["alt"],
                delta_bp=delta_bp,
                delta_cM=delta_cM,
                alt_freq_by_pop=it["alt_freq_by_pop"],
                score_af_range=it["score"],
            )
        )

    return out


# ---------------------------
# 3) End-to-end example
# ---------------------------

if __name__ == "__main__":
    # Parameters chosen to increase differentiation:
    # pick a target FST, compute an approximate pairwise migration rate m
    K = 5
    Ne = 5_000
    target_fst = 0.25
    m = pairwise_m_for_target_fst(target_fst=target_fst, Ne=Ne, K=K)

    # Simulate C chromosomes
    chroms = simulate_island_model(
        K=K,
        n_diploids_per_island=20,
        C=5,
        Ne=Ne,
        m=m,
        sequence_length=1e6,
        recombination_rate=1e-8,
        mutation_rate=1e-8,
        random_seed=123,
    )

    # Select top M markers and compute distances between them
    markers = top_markers_by_af_range(
        chroms,
        M=50,
        recombination_rate=1e-8,
        strict_single_mutation=True,
    )

    print("Number of chromosomes:", len(chroms))
    print("Markers returned:", len(markers))
    if markers:
        # Each record includes delta_bp and delta_cM
        print("Example marker record:", markers[0])

        # Show first few markers with distances
        for mrec in markers[:10]:
            print(
                f"chr{mrec.chrom}\tpos={mrec.pos_bp:.0f}\t"
                f"delta_bp={mrec.delta_bp}\tdelta_cM={mrec.delta_cM}\t"
                f"score={mrec.score_af_range:.3f}"
            )
