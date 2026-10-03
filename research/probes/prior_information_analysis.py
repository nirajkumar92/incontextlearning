"""Exact information diagnostics for synthetic-prior design, with CPU checks.

These are small, fully specified statistical experiments. They do not measure a
trained model, rank prior families, or establish a mixture's downstream value.
They show why a generator must jointly control mechanism, observations, and
context size. Run directly; NumPy is the only third-party dependency.

Normal random effects: theta ~ N(0, tau2), y_i | theta ~ N(theta, sigma2).
The posterior mean is a_n * mean(y), a_n = n*tau2/(sigma2+n*tau2), and its
integrated squared-error risk for theta is v_n = tau2*sigma2/(sigma2+n*tau2).

Occupancy: for independent category draws with probabilities p_j, the expected
probability mass of unseen categories is sum_j p_j*(1-p_j)**n. This differs
from the fraction of category identities unseen when p is nonuniform.

Affine parity versus lookup: both receive x uniform on {0,1}**k, but one has
y = b XOR <a,x> and a uniform prior on (a,b); the other has independent fair
labels for each of 2**k cells. Given a noiseless affine context of augmented
rank r >= 1, a uniform query is identified with probability 2**(r-1-k).
The lookup query is identified only when its cell was observed. Thus the same
occupancy does not imply the same learnability for different mechanism priors.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def normal_random_effects(n, tau2=1.0, sigma2=4.0):
    """Posterior shrinkage and exact Bayes risk; n=0 means use the prior mean."""
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    if not all(np.isfinite(x) and x > 0 for x in (tau2, sigma2)):
        raise ValueError("variances must be finite and positive")
    denominator = sigma2 + n * tau2
    variance = tau2 * sigma2 / denominator
    return {
        "support_count": int(n),
        "shrinkage_toward_sample_mean": float(n * tau2 / denominator),
        "posterior_theta_variance": float(variance),
        "bayes_theta_mse": float(variance),
        "bayes_new_response_mse": float(variance + sigma2),
        "unpooled_sample_mean_theta_mse": None if n == 0 else float(sigma2 / n),
        "prior_mean_theta_mse": float(tau2),
    }


def category_occupancy(probabilities, n):
    """Exact expectations for iid support and an independent same-law query."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not len(p) or not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be a nonempty finite vector")
    if np.any(p < 0) or not np.isclose(p.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("probabilities must be nonnegative and sum to one")
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    # Explicit n=0 handles deterministic categories without logarithmic 0/0.
    unseen = np.ones_like(p) if n == 0 else (1.0 - p) ** n
    return {
        "support_count": int(n),
        "category_count": len(p),
        "expected_unseen_query_mass": float(p @ unseen),
        "expected_distinct_observed_categories": float(np.sum(1.0 - unseen)),
        "expected_fraction_of_identities_unseen": float(np.mean(unseen)),
    }


def entropy(probabilities):
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def coarsened_observation():
    """Enumerate P(X,Y) after discrete independent sensor noise and quantizing.

    Z is uniform on (-2,-1,1,2), Y=1[Z>0], E in (-2,0,2) with probabilities
    (.2,.6,.2), W=Z+E, and X=0 if W<-1, 1 if -1<=W<=1, otherwise 2.
    The mechanism is known here. Support is irrelevant to these oracle bounds.
    """
    z_values = np.array([-2, -1, 1, 2])
    errors = np.array([-2, 0, 2])
    error_probabilities = np.array([0.2, 0.6, 0.2])
    joint = np.zeros((3, 2), dtype=float)
    raw_joint = {}
    for z in z_values:
        for error, probability in zip(errors, error_probabilities):
            w = int(z + error)
            x = 0 if w < -1 else (1 if w <= 1 else 2)
            y = int(z > 0)
            joint[x, y] += 0.25 * probability
            raw_joint.setdefault(w, np.zeros(2))[y] += 0.25 * probability
    mass = joint.sum(axis=1)
    posterior = joint[:, 1] / mass
    conditional_entropy = sum(float(row.sum()) * entropy(row / row.sum()) for row in joint)
    raw_conditional_entropy = sum(float(row.sum()) * entropy(row / row.sum()) for row in raw_joint.values())
    return {
        "z_values": z_values.tolist(),
        "z_probabilities": [0.25] * 4,
        "noise_values": errors.tolist(),
        "noise_probabilities": error_probabilities.tolist(),
        "quantizer": "0 if W < -1; 1 if -1 <= W <= 1; 2 if W > 1",
        "joint_X_Y": joint.tolist(),
        "observed_X_probabilities": mass.tolist(),
        "positive_probability_given_observed_X": posterior.tolist(),
        "bayes_classification_error": float(np.minimum(joint[:, 0], joint[:, 1]).sum()),
        "bayes_binary_brier_risk": float(np.sum(mass * posterior * (1.0 - posterior))),
        "bayes_log_loss_nats": float(conditional_entropy),
        "bayes_log_loss_observing_noisy_unquantized_W_nats": float(raw_conditional_entropy),
        "clairvoyant_latent_Z_log_loss_nats": 0.0,
        "limitation": "Known generative law, not an unknown-law PFN. Latent-conditioned certainty is not the observed-input predictive posterior. Latent soft labels can still be an unbiased loss-only target when correctly averaged over latent states.",
    }


def affine_parity_risk(k, n):
    """Exact Bayes error by propagating the augmented GF(2) design-rank law."""
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= 30:
        raise ValueError("k must be an integer in [1, 30]")
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    ranks = np.zeros(k + 2)
    ranks[0] = 1.0
    for _ in range(n):
        updated = np.zeros_like(ranks)
        for rank, probability in enumerate(ranks):
            if probability == 0:
                continue
            dependent = 0.0 if rank == 0 else 2.0 ** (rank - 1 - k)
            updated[rank] += probability * dependent
            if rank < k + 1:
                updated[rank + 1] += probability * (1.0 - dependent)
        ranks = updated
    unidentified = np.array([1.0] + [1.0 - 2.0 ** (r - 1 - k) for r in range(1, k + 2)])
    probability_unidentified = float(ranks @ unidentified)
    return {
        "feature_bits": int(k),
        "support_count": int(n),
        "augmented_rank_probabilities": ranks.tolist(),
        "affine_parity_query_unidentified_probability": probability_unidentified,
        "affine_parity_bayes_classification_error": 0.5 * probability_unidentified,
        "independent_truth_table_bayes_classification_error": 0.5 * (1.0 - 2.0 ** -k) ** n,
    }


def gf2_basis(rows):
    """Row basis with integer bit vectors, for a separate Monte Carlo check."""
    basis = {}
    for row in rows:
        value = int(row)
        while value:
            pivot = value.bit_length() - 1
            if pivot in basis:
                value ^= basis[pivot]
            else:
                basis[pivot] = value
                break
    return basis


def in_gf2_span(row, basis):
    value = int(row)
    while value:
        pivot = value.bit_length() - 1
        if pivot not in basis:
            return False
        value ^= basis[pivot]
    return True


def sample_summary(values, bernoulli=False):
    values = np.asarray(values, dtype=float)
    count = int(values.size)
    if count < 2:
        raise ValueError("at least two independent replicates are needed")
    mean = float(values.mean())
    result = {
        "independent_replicates": count,
        "mean": mean,
        "monte_carlo_standard_error": float(values.std(ddof=1) / math.sqrt(count)),
    }
    if bernoulli:
        # Wilson interval remains nondegenerate after zero observed failures.
        z = 1.959963984540054
        denominator = 1.0 + z * z / count
        center = (mean + z * z / (2.0 * count)) / denominator
        half = z * math.sqrt(mean * (1.0 - mean) / count + z * z / (4.0 * count * count)) / denominator
        result["wilson_95_percent_interval"] = [max(0.0, center - half), min(1.0, center + half)]
    return result


def run(seed=20261003, replicates=50000, occupancy_replicates=4000, parity_replicates=4000):
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if min(replicates, occupancy_replicates, parity_replicates) < 2:
        raise ValueError("all replicate counts must be at least two")
    streams = np.random.SeedSequence(seed).spawn(4)
    normal_rng, occupancy_rng, coarsening_rng, parity_rng = [np.random.default_rng(s) for s in streams]
    shrinkage_rows = []
    tau2, sigma2 = 1.0, 4.0
    for n in (0, 1, 2, 4, 16, 64, 256):
        row = normal_random_effects(n, tau2, sigma2)
        theta = normal_rng.normal(scale=math.sqrt(tau2), size=replicates)
        sample_mean = theta + normal_rng.normal(scale=math.sqrt(sigma2 / n), size=replicates) if n else np.zeros(replicates)
        prediction = row["shrinkage_toward_sample_mean"] * sample_mean
        row["simulated_theta_mse"] = sample_summary((prediction - theta) ** 2)
        row["simulated_response_mse"] = sample_summary((prediction - theta - normal_rng.normal(scale=math.sqrt(sigma2), size=replicates)) ** 2)
        shrinkage_rows.append(row)

    occupancy_rows = []
    k = 128
    zipf = np.arange(1, k + 1, dtype=float) ** -1.25
    for name, p in (("uniform", np.ones(k) / k), ("zipf_exponent_1.25", zipf / zipf.sum())):
        for n in (32, 128, 512):
            row = {"frequency_law": name, **category_occupancy(p, n)}
            counts = occupancy_rng.multinomial(n, p, size=occupancy_replicates)
            row["simulated_unseen_query_mass"] = sample_summary((counts == 0) @ p)
            row["simulated_distinct_observed_categories"] = sample_summary(np.sum(counts > 0, axis=1))
            occupancy_rows.append(row)

    coarsening = coarsened_observation()
    z = coarsening_rng.choice(coarsening["z_values"], size=replicates)
    y = (z > 0).astype(int)
    w = z + coarsening_rng.choice(coarsening["noise_values"], size=replicates, p=coarsening["noise_probabilities"])
    x = np.where(w < -1, 0, np.where(w <= 1, 1, 2))
    probabilities = np.array(coarsening["positive_probability_given_observed_X"])[x]
    coarsening["simulated_brier_risk"] = sample_summary((probabilities - y) ** 2)
    coarsening["simulated_log_loss_nats"] = sample_summary(-np.log(np.where(y, probabilities, 1.0 - probabilities)))
    coarsening["simulated_classification_error"] = sample_summary((probabilities >= 0.5) != y, bernoulli=True)

    interaction_rows = []
    k = 12
    for n in (0, 1, 4, 8, 12, 16, 32, 128):
        row = affine_parity_risk(k, n)
        unidentified, query_cells_unseen = [], []
        for _ in range(parity_replicates):
            rows = parity_rng.integers(0, 2 ** k, size=n)
            query = int(parity_rng.integers(0, 2 ** k))
            basis = gf2_basis(rows | (1 << k))
            unidentified.append(not in_gf2_span(query | (1 << k), basis))
            query_cells_unseen.append(not np.any(rows == query))
        row["simulated_affine_query_unidentified"] = sample_summary(unidentified, bernoulli=True)
        row["simulated_lookup_query_unseen"] = sample_summary(query_cells_unseen, bernoulli=True)
        interaction_rows.append(row)

    return {
        "schema_version": 1,
        "status": "Exact toy Bayes calculations and executed CPU Monte Carlo checks; no trained model or benchmark results.",
        "seed": seed,
        "rng": "NumPy PCG64; SeedSequence(seed).spawn(4) in section order",
        "numpy_version": np.__version__,
        "claim_limits": [
            "These calculations constrain generator design; they do not identify optimal mixture weights or predict a SOTA win.",
            "Bayes risks assume the explicitly specified generative law, noise, and latent-parameter prior are correct.",
            "Low synthetic Bayes risk does not establish useful real-data transfer or neural optimization efficiency.",
            "No fixed predictive-mixture regret bound is asserted for a posterior mixture of task-generating priors; posterior component weights depend on context evidence.",
        ],
        "normal_random_effects": {
            "tau_squared": tau2, "sigma_squared": sigma2,
            "assumptions": "Known zero population mean and known variances; one category effect. Hyperparameter estimation and category hierarchy are not simulated.",
            "rows": shrinkage_rows,
        },
        "category_occupancy": {
            "assumptions": "Independent support/query categories from the same fixed frequency law. Occupancy alone says nothing about cross-category sharing or target mechanism.",
            "rows": occupancy_rows,
        },
        "coarsened_observation": coarsening,
        "interaction_mechanism_counterexample": {
            "assumptions": "Uniform iid binary features, noiseless labels; uniform prior on affine-parity coefficients versus independent fair lookup labels. Both observe all feature bits.",
            "interpretation": "At fixed input occupancy, shared algebraic structure can make an interaction family learnable with far less support than an unstructured lookup family.",
            "rows": interaction_rows,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--replicates", type=int, default=50000)
    parser.add_argument("--occupancy-replicates", type=int, default=4000)
    parser.add_argument("--parity-replicates", type=int, default=4000)
    args = parser.parse_args()
    try:
        results = run(args.seed, args.replicates, args.occupancy_replicates, args.parity_replicates)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
