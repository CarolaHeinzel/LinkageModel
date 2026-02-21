"""End-to-end evaluation (diploid) using markers selected from an msprime island model.

Workflow
--------
1) Simulate a symmetric K-deme island model (diploid sampling) with msprime.
2) Select the top M markers by allele-frequency range across demes.
3) Build p (allele frequencies by deme) and d (inter-marker distances in cM).
4) Simulate diploid genotypes under
     - Admixture model (iid across markers)
     - Linkage model (HMM along markers)
   and re-run the LRT test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Sequence, Union, Tuple
import itertools
import math
import heapq

import numpy as np

# External dependency (not bundled here)
import msprime

import Simulation_Linkage as sL
import Test_LM_AM as test


# ---------------------------
# Helper: choose migration rate to hit a rough target Fst
# ---------------------------

def pairwise_m_for_target_fst(target_fst: float, Ne: float, K: int) -> float:
    """Rough island-model approximation:
        FST ≈ 1 / (1 + 4 * Ne * (K-1) * m_pairwise)
    Solve for m_pairwise.
    """
    if not (0 < target_fst < 1):
        raise ValueError("target_fst must be between 0 and 1.")
    if Ne <= 0:
        raise ValueError("Ne must be > 0.")
    if K < 2:
        return 0.0
    return (1.0 / target_fst - 1.0) / (4.0 * Ne * (K - 1))


# ---------------------------
# 1) Island model simulation (C chromosomes)
# ---------------------------

ChromosomeSet = List[msprime.TreeSequence]
TreeSeqOrIter = Union[
    msprime.TreeSequence,
    ChromosomeSet,
    Generator[ChromosomeSet, None, None],
]


def simulate_island_model(
    K: int,
    n_diploids_per_island: int,
    *,
    C: int = 1,
    Ne: float = 10_000,
    m: float = 1e-3,  # pairwise migration rate
    sequence_length: float = 2e7,
    recombination_rate: float = 1e-8,
    mutation_rate: float = 1e-7,
    dtwf_generations: Optional[int] = None,
    num_replicates: Optional[int] = None,
    random_seed: Optional[int] = 42,
) -> TreeSeqOrIter:
    """Simulate a symmetric K-deme island model and return C independent chromosomes.

    Note: sampling uses ploidy=2, but msprime returns haploid sample nodes.
    Allele frequencies are computed on haplotypes (which is standard for AF).
    """
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
    delta_bp: Optional[float]
    delta_cM: Optional[float]
    alt_freq_by_pop: Dict[int, float]
    score_af_range: float


def top_markers_by_af_range(
    chroms: TreeSeqOrSeq,
    M: int,
    *,
    recombination_rate: float,
    strict_single_mutation: bool = True,
) -> List[SNPRecord]:
    """Pick top-M biallelic polymorphic SNPs by AF range across populations.

    Distance approximation:
      delta_cM = 100 * recombination_rate * delta_bp
    """
    chrom_list = [chroms] if isinstance(chroms, msprime.TreeSequence) else list(chroms)

    heap: List[Tuple[float, int, dict]] = []
    tie_id = 0

    for chrom_idx, ts in enumerate(chrom_list):
        samples = ts.samples()
        node_pops = ts.tables.nodes.population[samples]
        pop_ids = sorted({int(p) for p in node_pops})

        pop_masks = {pid: (node_pops == pid) for pid in pop_ids}
        pop_denoms = {pid: int(mask.sum()) for pid, mask in pop_masks.items()}

        for var in ts.variants():
            if len(var.alleles) != 2:
                continue
            if strict_single_mutation and len(var.site.mutations) != 1:
                continue
            if any(a is None or len(a) != 1 for a in var.alleles):
                continue

            g = var.genotypes
            if np.any(g < 0) or np.any(g > 1):
                continue

            alt_count = int(g.sum())
            n = len(g)
            if alt_count == 0 or alt_count == n:
                continue

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

    selected = [t[2] for t in heap]
    selected.sort(key=lambda x: (x["chrom"], x["pos_bp"]))

    prev_pos_by_chrom: Dict[int, float] = {}
    out: List[SNPRecord] = []

    for it in selected:
        chrom = it["chrom"]
        pos = it["pos_bp"]

        if chrom in prev_pos_by_chrom:
            delta_bp = pos - prev_pos_by_chrom[chrom]
            delta_cM = 100.0 * recombination_rate * delta_bp
        else:
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


def build_p_and_d_from_markers(
    markers: List[SNPRecord],
    K: int,
    *,
    eps: float = 1e-4,
    chrom_break_cM: float = 1e6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct p (K,M) and d (M,) from selected marker records."""
    M = len(markers)
    if M == 0:
        raise ValueError("No markers provided.")

    p = np.zeros((K, M), dtype=float)
    d = np.zeros(M, dtype=float)

    for t, rec in enumerate(markers):
        for k in range(K):
            p[k, t] = float(rec.alt_freq_by_pop.get(k, math.nan))

        if t == 0:
            d[t] = 0.0
        else:
            if rec.delta_cM is None:
                d[t] = float(chrom_break_cM)  # break between chromosomes
            else:
                d[t] = float(rec.delta_cM)

    # clip p away from 0/1 to avoid -inf likelihoods
    p = np.clip(p, eps, 1.0 - eps)
    return p, d


# ---------------------------
# 3) Evaluate the LRT using diploid individuals
# ---------------------------


def evaluate_once(p: np.ndarray, d: np.ndarray, q_true: np.ndarray, r_true: float, *, rng) -> Tuple[bool, bool]:
    """Return (accept_H0_under_AM_truth, reject_H0_under_LM_truth)."""
    K, M = p.shape

    # Admixture truth
    X_am = test.create_sample_pbekannt_diploid(M, K, p, q_true, rng=rng)
    res_am = test.test_summary(X_am, p, d, K, ploidy=2, n_em_iter=80, lm_maxiter=40)
    accept_am = (res_am["reject_H0"] is False)

    # Linkage truth
    X_lm, _ = sL.simulate_markov_chain_diploid(K, q_true, r_true, M, d, p, rng=rng)
    res_lm = test.test_summary(X_lm, p, d, K, ploidy=2, n_em_iter=80, lm_maxiter=40)
    reject_lm = (res_lm["reject_H0"] is True)

    return accept_am, reject_lm


def evaluate_replicates(p: np.ndarray, d: np.ndarray, *, r_true: float, num_rep: int = 50, seed: int = 0) -> Tuple[float, float]:
    """Return (accept rate under AM truth, power under LM truth)."""
    rng = np.random.default_rng(seed)
    K, _M = p.shape

    accept = 0
    power = 0

    for i in range(num_rep):
        # random admixture per individual
        q_true = rng.dirichlet(np.ones(K))

        a, b = evaluate_once(p, d, q_true, r_true, rng=rng)
        accept += int(a)
        power += int(b)

        if (i + 1) % 10 == 0:
            print(f"rep {i+1}/{num_rep}  accept_AM={accept/(i+1):.3f}  power_LM={power/(i+1):.3f}")

    return accept / num_rep, power / num_rep


# ---------------------------
# 4) Main
# ---------------------------

if __name__ == "__main__":
    # --- Island model parameters ---
    K = 5
    Ne = 5_000
    target_fst = 0.25
    m = pairwise_m_for_target_fst(target_fst=target_fst, Ne=Ne, K=K)

    # You may need to increase sequence_length / mutation_rate if you get < 1000 SNPs.
    C = 5
    sequence_length = 2e7
    recombination_rate = 1e-8
    mutation_rate = 1e-7

    # --- Marker selection ---
    M = 1000
    chroms = simulate_island_model(
        K=K,
        n_diploids_per_island=20,
        C=C,
        Ne=Ne,
        m=m,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        mutation_rate=mutation_rate,
        random_seed=123,
    )

    markers = top_markers_by_af_range(
        chroms,
        M=M,
        recombination_rate=recombination_rate,
        strict_single_mutation=True,
    )

    if len(markers) < M:
        raise RuntimeError(
            f"Only {len(markers)} polymorphic SNPs passed filters; need M={M}. "
            "Increase sequence_length and/or mutation_rate (or relax strict_single_mutation)."
        )

    p, d = build_p_and_d_from_markers(markers, K)

    print(f"Selected markers: {len(markers)}")
    print(f"p shape: {p.shape}, d shape: {d.shape}")
    print("Example d[:10] (cM):", d[:10])

    # --- Test evaluation ---
    for r_true in [0.1, 1.0, 10.0]:
        acc, powr = evaluate_replicates(p, d, r_true=r_true, num_rep=20, seed=0)
        print(f"r_true={r_true}  accept_AM={acc:.3f}  power_LM={powr:.3f}")
