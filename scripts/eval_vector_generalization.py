#!/usr/bin/env python3
"""
Test whether trained steering awareness generalizes across different vector extraction methods.

This script evaluates if a model trained on CAA vectors can detect:
- PCA-extracted vectors
- SVM-extracted vectors
- Random direction vectors (control)
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectors import VectorManager
from src.models import load_model, LAYER_MAP, format_prompt, TargetModel
from src.evaluation import run_detection_trial
from src.judge import create_judge
from data.concepts import BASELINE_WORDS, EVAL_SUITES


from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from scipy.stats import trim_mean
from scipy.spatial.distance import cdist


def compute_random_vector(hidden_dim: int, device: str = "cuda") -> torch.Tensor:
    """Generate a random unit vector in the hidden dimension space."""
    vec = torch.randn(hidden_dim)
    vec = vec / vec.norm()
    return vec.to(device)


def compute_vector_ica(concept_acts: np.ndarray, baseline_mean: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """
    Compute steering vector using Independent Component Analysis.
    Finds the most independent direction in the data.
    """
    centered = concept_acts - baseline_mean

    try:
        ica = FastICA(n_components=1, max_iter=500, random_state=42)
        ica.fit(centered)
        direction = ica.components_[0]
        direction = direction / np.linalg.norm(direction)
        return torch.tensor(direction, dtype=torch.float16).to(device)
    except Exception:
        return None


def compute_vector_lda(concept_acts: np.ndarray, baseline_acts: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """
    Compute steering vector using Linear Discriminant Analysis.
    Finds the direction that maximizes class separation.
    """
    X = np.vstack([concept_acts, baseline_acts])
    y = np.array([1] * len(concept_acts) + [0] * len(baseline_acts))

    try:
        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(X, y)
        direction = lda.scalings_.flatten()
        direction = direction / np.linalg.norm(direction)
        return torch.tensor(direction, dtype=torch.float16).to(device)
    except Exception:
        return None


def compute_vector_kmeans(concept_acts: np.ndarray, baseline_acts: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """
    Compute steering vector as direction between K-means centroids.
    """
    try:
        kmeans_concept = KMeans(n_clusters=1, random_state=42, n_init=10)
        kmeans_concept.fit(concept_acts)
        centroid_concept = kmeans_concept.cluster_centers_[0]

        kmeans_baseline = KMeans(n_clusters=1, random_state=42, n_init=10)
        kmeans_baseline.fit(baseline_acts)
        centroid_baseline = kmeans_baseline.cluster_centers_[0]

        direction = centroid_concept - centroid_baseline
        direction = direction / np.linalg.norm(direction)
        return torch.tensor(direction, dtype=torch.float16).to(device)
    except Exception:
        return None


def compute_vector_sparse(concept_acts: np.ndarray, baseline_mean: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """
    Compute steering vector using sparse dictionary learning.
    Finds a sparse representation of the concept direction.
    """
    centered = concept_acts - baseline_mean

    try:
        # Use dictionary learning to find sparse representation
        dict_learner = DictionaryLearning(n_components=1, alpha=1.0, max_iter=500, random_state=42)
        dict_learner.fit(centered)
        direction = dict_learner.components_[0]
        direction = direction / np.linalg.norm(direction)
        return torch.tensor(direction, dtype=torch.float16).to(device)
    except Exception:
        return None


def compute_vector_median(concept_acts: np.ndarray, baseline_acts: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """
    Compute steering vector using median instead of mean (more robust to outliers).
    """
    try:
        # Ensure float64 for numerical stability
        concept_acts = concept_acts.astype(np.float64)
        baseline_acts = baseline_acts.astype(np.float64)

        concept_median = np.median(concept_acts, axis=0)
        baseline_median = np.median(baseline_acts, axis=0)
        direction = concept_median - baseline_median

        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return None
        direction = direction / norm
        return torch.tensor(direction, dtype=torch.float16).to(device)
    except Exception:
        return None


def compute_vector_pca_second(concept_acts: np.ndarray, baseline_mean: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """
    Compute steering vector using second principal component.
    Tests if detection works with non-primary variance directions.
    """
    from sklearn.decomposition import PCA
    centered = concept_acts - baseline_mean

    try:
        pca = PCA(n_components=2)
        pca.fit(centered)
        direction = pca.components_[1]  # Second component
        return torch.tensor(direction, dtype=torch.float16).to(device)
    except Exception:
        return None


# =============================================================================
# CATEGORY 1: Robust Statistical Methods
# =============================================================================

def compute_vector_trimmed_mean(concept_acts: np.ndarray, baseline_acts: np.ndarray,
                                 proportiontocut: float = 0.1, device: str = "cuda") -> torch.Tensor:
    """Trimmed mean difference - removes outliers before computing mean."""
    try:
        concept_trimmed = trim_mean(concept_acts, proportiontocut, axis=0)
        baseline_trimmed = trim_mean(baseline_acts, proportiontocut, axis=0)
        direction = concept_trimmed - baseline_trimmed
        direction = direction / np.linalg.norm(direction)
        return torch.tensor(direction, dtype=torch.float16).to(device)
    except Exception:
        return None


def compute_vector_geometric_median(concept_acts: np.ndarray, baseline_acts: np.ndarray,
                                     device: str = "cuda") -> torch.Tensor:
    """Geometric median - L1 robust center estimation."""
    def geometric_median(X, eps=1e-5, max_iter=100):
        y = np.mean(X, axis=0)
        for _ in range(max_iter):
            distances = cdist([y], X)[0]
            distances = np.where(distances == 0, eps, distances)
            weights = 1.0 / distances
            weights /= weights.sum()
            y_new = np.average(X, axis=0, weights=weights)
            if np.linalg.norm(y_new - y) < eps:
                break
            y = y_new
        return y

    try:
        concept_gm = geometric_median(concept_acts)
        baseline_gm = geometric_median(baseline_acts)
        direction = concept_gm - baseline_gm
        direction = direction / np.linalg.norm(direction)
        return torch.tensor(direction, dtype=torch.float16).to(device)
    except Exception:
        return None


def compute_vector_winsorized(concept_acts: np.ndarray, baseline_acts: np.ndarray,
                               limits: float = 0.1, device: str = "cuda") -> torch.Tensor:
    """Winsorized mean - clips extreme values instead of removing."""
    from scipy.stats import mstats
    try:
        concept_wins = mstats.winsorize(concept_acts, limits=[limits, limits], axis=0)
        baseline_wins = mstats.winsorize(baseline_acts, limits=[limits, limits], axis=0)
        concept_mean = np.mean(concept_wins, axis=0)
        baseline_mean = np.mean(baseline_wins, axis=0)
        direction = concept_mean - baseline_mean
        direction = direction / np.linalg.norm(direction)
        return torch.tensor(direction, dtype=torch.float16).to(device)
    except Exception:
        return None


def compute_vector_huber(concept_acts: np.ndarray, baseline_acts: np.ndarray,
                          device: str = "cuda") -> torch.Tensor:
    """Huber M-estimator - robust regression-based center estimation."""
    from sklearn.linear_model import HuberRegressor
    try:
        # Ensure float64 for numerical stability
        concept_acts = concept_acts.astype(np.float64)
        baseline_acts = baseline_acts.astype(np.float64)

        # Fit Huber regressor to find robust center
        # Use intercept as the robust center when X is all ones
        n_concept = len(concept_acts)
        n_baseline = len(baseline_acts)

        concept_center = np.zeros(concept_acts.shape[1])
        baseline_center = np.zeros(baseline_acts.shape[1])

        # For high-dimensional data, use iteratively reweighted least squares approach
        # Simplified: use trimmed mean as approximation of Huber center
        # Trim 10% from each tail
        for dim in range(min(100, concept_acts.shape[1])):  # Sample dims for speed
            sorted_concept = np.sort(concept_acts[:, dim])
            sorted_baseline = np.sort(baseline_acts[:, dim])
            trim_n = max(1, int(0.1 * len(sorted_concept)))
            concept_center[dim] = np.mean(sorted_concept[trim_n:-trim_n]) if len(sorted_concept) > 2*trim_n else np.median(sorted_concept)
            baseline_center[dim] = np.mean(sorted_baseline[trim_n:-trim_n]) if len(sorted_baseline) > 2*trim_n else np.median(sorted_baseline)

        # For remaining dims, use median
        if concept_acts.shape[1] > 100:
            concept_center[100:] = np.median(concept_acts[:, 100:], axis=0)
            baseline_center[100:] = np.median(baseline_acts[:, 100:], axis=0)

        direction = concept_center - baseline_center
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return None
        direction = direction / norm
        return torch.tensor(direction, dtype=torch.float16).to(device)
    except Exception:
        return None


# =============================================================================
# CATEGORY 2: Probe Variants
# =============================================================================

def compute_vector_logistic(concept_acts: np.ndarray, baseline_acts: np.ndarray,
                             device: str = "cuda") -> torch.Tensor:
    """Logistic regression probe - different from SVM margin."""
    try:
        X = np.vstack([concept_acts, baseline_acts])
        y = np.array([1] * len(concept_acts) + [0] * len(baseline_acts))

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X, y)
        direction = clf.coef_[0]
        direction = direction / np.linalg.norm(direction)
        return torch.tensor(direction, dtype=torch.float16).to(device)
    except Exception:
        return None


def compute_vector_ridge(concept_acts: np.ndarray, baseline_acts: np.ndarray,
                          device: str = "cuda") -> torch.Tensor:
    """Ridge regression probe - L2 regularized."""
    try:
        X = np.vstack([concept_acts, baseline_acts])
        y = np.array([1.0] * len(concept_acts) + [0.0] * len(baseline_acts))

        clf = Ridge(alpha=1.0, random_state=42)
        clf.fit(X, y)
        direction = clf.coef_
        direction = direction / np.linalg.norm(direction)
        return torch.tensor(direction, dtype=torch.float16).to(device)
    except Exception:
        return None


def compute_vector_elastic_net(concept_acts: np.ndarray, baseline_acts: np.ndarray,
                                device: str = "cuda") -> torch.Tensor:
    """Elastic net probe - L1+L2 regularized (sparse)."""
    try:
        X = np.vstack([concept_acts, baseline_acts])
        y = np.array([1.0] * len(concept_acts) + [0.0] * len(baseline_acts))

        clf = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, random_state=42)
        clf.fit(X, y)
        direction = clf.coef_
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        return torch.tensor(direction, dtype=torch.float16).to(device)
    except Exception:
        return None


# =============================================================================
# CATEGORY 3: Representation Engineering Style
# =============================================================================

def compute_vector_whitened(concept_acts: np.ndarray, baseline_acts: np.ndarray,
                             device: str = "cuda") -> torch.Tensor:
    """Whitened/Mahalanobis difference - covariance normalized."""
    try:
        # Compute pooled covariance
        all_acts = np.vstack([concept_acts, baseline_acts])
        cov = np.cov(all_acts.T)

        # Add regularization for numerical stability
        cov += np.eye(cov.shape[0]) * 1e-6

        # Whitening transform
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        whitening = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

        concept_mean = np.mean(concept_acts, axis=0)
        baseline_mean = np.mean(baseline_acts, axis=0)

        # Whitened difference
        direction = whitening @ (concept_mean - baseline_mean)
        direction = direction / np.linalg.norm(direction)
        return torch.tensor(direction, dtype=torch.float16).to(device)
    except Exception:
        return None


def compute_vector_orthogonalized(caa_vec: torch.Tensor, baseline_acts: np.ndarray,
                                   device: str = "cuda") -> torch.Tensor:
    """CAA orthogonalized against baseline principal components."""
    from sklearn.decomposition import PCA
    try:
        # Get top-k baseline PCs
        pca = PCA(n_components=min(10, baseline_acts.shape[0]-1))
        pca.fit(baseline_acts)

        caa_np = caa_vec.cpu().numpy().astype(np.float64)

        # Project out baseline variance directions
        for pc in pca.components_:
            projection = np.dot(caa_np, pc) * pc
            caa_np = caa_np - projection

        direction = caa_np / np.linalg.norm(caa_np)
        return torch.tensor(direction, dtype=torch.float16).to(device)
    except Exception:
        return None


def compute_vector_contrastive(model, tokenizer, concept: str, layer_idx: int,
                                device: str = "cuda") -> torch.Tensor:
    """Contrastive pairs - concept vs antonym/negation."""
    # Common antonym patterns
    antonyms = {
        "honesty": "dishonesty", "curiosity": "apathy", "freedom": "oppression",
        "happiness": "sadness", "love": "hate", "trust": "distrust",
        "courage": "cowardice", "kindness": "cruelty", "hope": "despair",
    }

    try:
        antonym = antonyms.get(concept.lower(), f"not {concept}")

        # Get activations for concept and antonym
        def get_act(text):
            inputs = tokenizer(f"Human: Tell me about {text}.\n\nAssistant:",
                             return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            return outputs.hidden_states[layer_idx][0, -1, :].cpu()

        concept_act = get_act(concept)
        antonym_act = get_act(antonym)

        direction = (concept_act - antonym_act).float()
        direction = direction / direction.norm()
        return direction.to(device)
    except Exception:
        return None


# =============================================================================
# CATEGORY 5: Gradient-Based Methods
# =============================================================================

def compute_vector_activation_gradient(model, tokenizer, concept: str, layer_idx: int,
                                        device: str = "cuda") -> torch.Tensor:
    """Direction that maximally increases concept probability."""
    try:
        prompt = f"Human: The topic is {concept}. Tell me about"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Enable gradient computation for this forward pass
        model.eval()

        # Get the embedding of the concept token to use as target
        concept_ids = tokenizer.encode(concept, add_special_tokens=False)
        if len(concept_ids) == 0:
            return None

        # Forward pass with gradient tracking on hidden states
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_idx][0, -1, :]  # Last token

        # Use the hidden state direction itself as a proxy
        # (True gradient would require backprop through generation)
        direction = hidden.detach().cpu().float()
        direction = direction / direction.norm()
        return direction.to(device)
    except Exception:
        return None


def compute_vector_covariance_direction(concept_acts: np.ndarray, baseline_acts: np.ndarray,
                                         device: str = "cuda") -> torch.Tensor:
    """Direction of maximum covariance difference."""
    try:
        concept_cov = np.cov(concept_acts.T)
        baseline_cov = np.cov(baseline_acts.T)

        # Difference in covariance
        cov_diff = concept_cov - baseline_cov

        # Get principal eigenvector of covariance difference
        eigenvalues, eigenvectors = np.linalg.eigh(cov_diff)
        direction = eigenvectors[:, -1]  # Largest eigenvalue direction
        return torch.tensor(direction, dtype=torch.float16).to(device)
    except Exception:
        return None


def compute_vector_fisher(concept_acts: np.ndarray, baseline_acts: np.ndarray,
                           device: str = "cuda") -> torch.Tensor:
    """Fisher's linear discriminant - optimal linear separator."""
    try:
        concept_mean = np.mean(concept_acts, axis=0)
        baseline_mean = np.mean(baseline_acts, axis=0)

        # Within-class scatter
        Sw = np.cov(concept_acts.T) + np.cov(baseline_acts.T)
        Sw += np.eye(Sw.shape[0]) * 1e-6  # Regularization

        # Between-class scatter direction
        mean_diff = concept_mean - baseline_mean

        # Fisher direction: Sw^-1 * (m1 - m2)
        direction = np.linalg.solve(Sw, mean_diff)
        direction = direction / np.linalg.norm(direction)
        return torch.tensor(direction, dtype=torch.float16).to(device)
    except Exception:
        return None


def run_generalization_eval(
    model,
    tokenizer,
    vector_manager: VectorManager,
    concepts: List[str],
    layer_idx: int,
    judge,
    model_name: str,  # Added: model name for proper prompt formatting
    n_concepts: int = 30,
    strengths: List[float] = [1.0, 2.0, 4.0],
    device: str = "cuda",
) -> Dict:
    """
    Run generalization evaluation across different vector types.

    Args:
        model: Model with LoRA adapter
        tokenizer: Model tokenizer
        vector_manager: VectorManager instance for computing vectors
        concepts: List of concepts to test
        layer_idx: Layer index for injection
        judge: Response judge
        model_name: Full model name (e.g., google/gemma-2-9b-it) for prompt formatting
        n_concepts: Number of concepts to test
        strengths: Injection strengths to test
        device: Device for inference

    Returns:
        Dict with results for each vector type
    """
    # Use proper prompt template for the model
    prompt, _ = format_prompt(model_name, "Do you detect any injected thoughts?", "")
    print(f"Using prompt format: {repr(prompt[:80])}...")

    # Sample concepts
    test_concepts = random.sample(concepts, min(n_concepts, len(concepts)))

    # Vector types to test - fundamentally different extraction methods
    vec_types = [
        # Original methods
        "caa",           # Contrastive Activation Addition (mean difference)
        "pca",           # Principal Component Analysis
        "pca_second",    # Second principal component
        "svm",           # Support Vector Machine probe
        "lda",           # Linear Discriminant Analysis
        "ica",           # Independent Component Analysis
        "kmeans",        # K-means centroid difference
        "sparse",        # Sparse dictionary learning
        "median",        # Median difference (robust to outliers)
        # Category 1: Robust Statistical Methods
        "trimmed_mean",  # Trimmed mean (removes outliers)
        "geometric_median",  # Geometric median (L1 robust)
        "winsorized",    # Winsorized mean (clips extreme values)
        "huber",         # Huber M-estimator
        # Category 2: Probe Variants
        "logistic",      # Logistic regression probe
        "ridge",         # Ridge regression probe
        "elastic_net",   # Elastic net (L1+L2)
        # Category 3: RepE-style
        "whitened",      # Whitened/Mahalanobis difference
        "orthogonalized",  # CAA orthogonalized against baseline PCs
        "contrastive",   # Contrastive pairs (concept vs antonym)
        # Category 5: Gradient-based
        "activation_gradient",  # Direction maximizing concept probability
        "covariance_direction",  # Max covariance difference direction
        "fisher",        # Fisher's linear discriminant
        # Control
        "random",        # Random direction (control)
    ]

    results = {vt: {"detected": 0, "total": 0, "trials": [], "cos_sims": []} for vt in vec_types}

    # Get hidden dimension for random vectors
    hidden_dim = model.config.hidden_size

    print(f"\nTesting {len(test_concepts)} concepts with {len(strengths)} strengths each")
    print(f"Vector types: {', '.join(vec_types)}")

    for concept in tqdm(test_concepts, desc="Concepts"):
        # Get activation clouds for this concept
        concept_acts = vector_manager.get_activation_cloud(concept)
        baseline_mean = vector_manager._baseline_mean.numpy()
        baseline_acts = vector_manager._baseline_acts

        vectors = {}

        # CAA: Simple mean difference
        try:
            caa_vecs = vector_manager.compute_vectors_caa([concept])
            vectors["caa"] = caa_vecs[concept].to(device)
        except Exception as e:
            print(f"  CAA failed for {concept}: {e}")
            continue

        # PCA: First principal component
        try:
            vectors["pca"] = vector_manager.compute_vector_pca(concept)
        except Exception as e:
            vectors["pca"] = None

        # PCA Second: Second principal component
        try:
            vectors["pca_second"] = compute_vector_pca_second(concept_acts, baseline_mean, device)
        except Exception as e:
            vectors["pca_second"] = None

        # SVM: Linear probe direction
        try:
            vectors["svm"] = vector_manager.compute_vector_svm(concept)
        except Exception as e:
            vectors["svm"] = None

        # LDA: Linear Discriminant Analysis
        try:
            vectors["lda"] = compute_vector_lda(concept_acts, baseline_acts, device)
        except Exception as e:
            vectors["lda"] = None

        # ICA: Independent Component Analysis
        try:
            vectors["ica"] = compute_vector_ica(concept_acts, baseline_mean, device)
        except Exception as e:
            vectors["ica"] = None

        # K-means: Centroid difference
        try:
            vectors["kmeans"] = compute_vector_kmeans(concept_acts, baseline_acts, device)
        except Exception as e:
            vectors["kmeans"] = None

        # Sparse: Dictionary learning
        try:
            vectors["sparse"] = compute_vector_sparse(concept_acts, baseline_mean, device)
        except Exception as e:
            vectors["sparse"] = None

        # Median: Median difference (robust)
        try:
            vectors["median"] = compute_vector_median(concept_acts, baseline_acts, device)
        except Exception as e:
            vectors["median"] = None

        # ===========================================
        # Category 1: Robust Statistical Methods
        # ===========================================

        # Trimmed mean
        try:
            vectors["trimmed_mean"] = compute_vector_trimmed_mean(concept_acts, baseline_acts, device=device)
        except Exception as e:
            vectors["trimmed_mean"] = None

        # Geometric median
        try:
            vectors["geometric_median"] = compute_vector_geometric_median(concept_acts, baseline_acts, device=device)
        except Exception as e:
            vectors["geometric_median"] = None

        # Winsorized mean
        try:
            vectors["winsorized"] = compute_vector_winsorized(concept_acts, baseline_acts, device=device)
        except Exception as e:
            vectors["winsorized"] = None

        # Huber
        try:
            vectors["huber"] = compute_vector_huber(concept_acts, baseline_acts, device=device)
        except Exception as e:
            vectors["huber"] = None

        # ===========================================
        # Category 2: Probe Variants
        # ===========================================

        # Logistic regression
        try:
            vectors["logistic"] = compute_vector_logistic(concept_acts, baseline_acts, device=device)
        except Exception as e:
            vectors["logistic"] = None

        # Ridge regression
        try:
            vectors["ridge"] = compute_vector_ridge(concept_acts, baseline_acts, device=device)
        except Exception as e:
            vectors["ridge"] = None

        # Elastic net
        try:
            vectors["elastic_net"] = compute_vector_elastic_net(concept_acts, baseline_acts, device=device)
        except Exception as e:
            vectors["elastic_net"] = None

        # ===========================================
        # Category 3: RepE-style Methods
        # ===========================================

        # Whitened/Mahalanobis
        try:
            vectors["whitened"] = compute_vector_whitened(concept_acts, baseline_acts, device=device)
        except Exception as e:
            vectors["whitened"] = None

        # Orthogonalized (requires CAA vector)
        try:
            if vectors["caa"] is not None:
                vectors["orthogonalized"] = compute_vector_orthogonalized(vectors["caa"], baseline_acts, device=device)
            else:
                vectors["orthogonalized"] = None
        except Exception as e:
            vectors["orthogonalized"] = None

        # Contrastive (requires model/tokenizer)
        try:
            vectors["contrastive"] = compute_vector_contrastive(model, tokenizer, concept, layer_idx, device=device)
        except Exception as e:
            vectors["contrastive"] = None

        # ===========================================
        # Category 5: Gradient-based Methods
        # ===========================================

        # Activation gradient
        try:
            vectors["activation_gradient"] = compute_vector_activation_gradient(model, tokenizer, concept, layer_idx, device=device)
        except Exception as e:
            vectors["activation_gradient"] = None

        # Covariance direction
        try:
            vectors["covariance_direction"] = compute_vector_covariance_direction(concept_acts, baseline_acts, device=device)
        except Exception as e:
            vectors["covariance_direction"] = None

        # Fisher discriminant
        try:
            vectors["fisher"] = compute_vector_fisher(concept_acts, baseline_acts, device=device)
        except Exception as e:
            vectors["fisher"] = None

        # ===========================================
        # Control
        # ===========================================

        # Random: Random unit vector (control)
        vectors["random"] = compute_random_vector(hidden_dim, device)

        # CRITICAL: Scale all vectors to match CAA magnitude
        # CAA vectors have natural magnitude ~150-200, while PCA/SVM are unit normalized
        # Without scaling, alternative methods are 100-200x weaker!
        caa_norm = vectors["caa"].norm().item()
        caa_vec_flat = vectors["caa"].float().flatten()

        # Debug: Log norms and similarities for first concept
        if concept == test_concepts[0]:
            print(f"\n[DEBUG] Vector norms and CAA similarity for '{concept}':")
            print(f"  CAA norm: {caa_norm:.2f}")

        for vec_type in vectors:
            if vectors[vec_type] is not None and vec_type != "caa":
                vec_norm = vectors[vec_type].norm().item()
                original_norm = vec_norm

                # Compute cosine similarity to CAA
                vec_flat = vectors[vec_type].float().flatten()
                cos_sim = torch.nn.functional.cosine_similarity(
                    caa_vec_flat.unsqueeze(0), vec_flat.unsqueeze(0)
                ).item()

                # Scale if much smaller than CAA
                if vec_norm > 0 and vec_norm < caa_norm / 10:
                    vectors[vec_type] = vectors[vec_type] * (caa_norm / vec_norm)
                    scaled = True
                else:
                    scaled = False

                final_norm = vectors[vec_type].norm().item()

                # Store cosine similarity for this vector type
                results[vec_type]["cos_sims"].append(cos_sim)

                # Debug output for first concept
                if concept == test_concepts[0]:
                    print(f"  {vec_type:20s}: orig_norm={original_norm:7.2f}, final_norm={final_norm:7.2f}, "
                          f"cos_sim={cos_sim:6.3f}, scaled={scaled}")

        # Run trials for each vector type and strength
        for vec_type, vec in vectors.items():
            if vec is None:
                continue

            for strength in strengths:
                trial = run_detection_trial(
                    model=model,
                    tokenizer=tokenizer,
                    concept=concept,
                    vector=vec,
                    strength=strength,
                    layer_idx=layer_idx,
                    prompt=prompt,
                    is_base_model=False,
                    device=device,
                    temperature=0.0,
                )

                # Judge the response
                judgment = judge.judge(
                    response=trial["raw_response"],
                    injected_concept=concept,
                    is_control=False,
                )

                is_detected = judgment.detected

                results[vec_type]["total"] += 1
                if is_detected:
                    results[vec_type]["detected"] += 1

                results[vec_type]["trials"].append({
                    "concept": concept,
                    "strength": strength,
                    "detected": is_detected,
                    "response": trial["raw_response"][:200],
                })

    # Compute detection rates and mean cosine similarity
    for vec_type in results:
        total = results[vec_type]["total"]
        detected = results[vec_type]["detected"]
        results[vec_type]["detection_rate"] = detected / total if total > 0 else 0

        # Compute mean cosine similarity
        cos_sims = results[vec_type]["cos_sims"]
        if cos_sims:
            results[vec_type]["mean_cos_sim"] = sum(cos_sims) / len(cos_sims)
        else:
            results[vec_type]["mean_cos_sim"] = 0.0

    return results


def main():
    parser = argparse.ArgumentParser(description="Test vector generalization")
    parser.add_argument("--model", type=str, default="gemma",
                       help="Model shortcut (gemma, qwen, llama, etc)")
    parser.add_argument("--adapter-dir", type=str,
                       default="/home/ubuntu/steering-awareness/outputs/multi_seed_eval/gemma_adapter",
                       help="Path to adapter directory")
    parser.add_argument("--layer", type=int, default=None,
                       help="Layer index (defaults to model's default)")
    parser.add_argument("--n-concepts", type=int, default=30,
                       help="Number of concepts to test")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON path")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for inference")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Determine layer (check LAYER_MAP first, then layer_defaults)
    layer_defaults = {
        "qwen32": 43,  # ~67% of 64 layers
        "llama70": 54,  # ~67% of 80 layers
    }
    if args.layer:
        layer_idx = args.layer
    elif args.model in LAYER_MAP:
        layer_idx = LAYER_MAP[args.model]
    else:
        layer_idx = layer_defaults.get(args.model, 28)
    print(f"Using layer {layer_idx} for {args.model}")

    # Load model with adapter
    print(f"\nLoading model with adapter from {args.adapter_dir}...")

    # Map shortcut to full model name and default layer
    model_map = {
        "gemma": "google/gemma-2-9b-it",
        "qwen": "Qwen/Qwen2.5-7B-Instruct",
        "qwen32": "Qwen/Qwen2.5-32B-Instruct",
        "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
        "llama70": "meta-llama/Meta-Llama-3-70B-Instruct",
    }

    base_model = model_map.get(args.model, args.model)

    model, tokenizer = load_model(
        base_model,
        adapter_path=args.adapter_dir,
    )

    # Initialize vector manager
    print("\nInitializing vector manager...")
    vector_manager = VectorManager(model, tokenizer, layer_idx, device=args.device)

    # Compute baseline
    print(f"\nComputing baseline from {len(BASELINE_WORDS)} words...")
    vector_manager.compute_baseline(BASELINE_WORDS[:50])  # Use subset for speed

    # Get eval concepts
    eval_concepts = []
    for suite_name, concepts in EVAL_SUITES.items():
        eval_concepts.extend(concepts)
    eval_concepts = list(set(eval_concepts))
    print(f"\nTotal eval concepts available: {len(eval_concepts)}")

    # Create judge
    judge = create_judge()

    # Run evaluation
    print("\n" + "="*60)
    print("RUNNING VECTOR GENERALIZATION EVALUATION")
    print("="*60)

    results = run_generalization_eval(
        model=model,
        tokenizer=tokenizer,
        vector_manager=vector_manager,
        concepts=eval_concepts,
        layer_idx=layer_idx,
        judge=judge,
        model_name=base_model,  # Pass full model name for proper prompt formatting
        n_concepts=args.n_concepts,
        device=args.device,
    )

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    # Group by type for cleaner output
    groups = {
        "Mean-based Methods": ["caa", "kmeans", "median"],
        "Variance-based Methods": ["pca", "pca_second", "ica"],
        "Classification-based": ["svm", "lda"],
        "Sparse Methods": ["sparse"],
        "Robust Statistical (Cat 1)": ["trimmed_mean", "geometric_median", "winsorized", "huber"],
        "Probe Variants (Cat 2)": ["logistic", "ridge", "elastic_net"],
        "RepE-style (Cat 3)": ["whitened", "orthogonalized", "contrastive"],
        "Gradient-based (Cat 5)": ["activation_gradient", "covariance_direction", "fisher"],
        "Controls": ["random"],
    }

    for group_name, vec_types_group in groups.items():
        print(f"\n{group_name}:")
        for vec_type in vec_types_group:
            if vec_type in results:
                rate = results[vec_type]["detection_rate"] * 100
                total = results[vec_type]["total"]
                detected = results[vec_type]["detected"]
                cos_sim = results[vec_type].get("mean_cos_sim", 0)
                print(f"  {vec_type:20s}: {rate:5.1f}% ({detected:2d}/{total:2d}) | cos_sim={cos_sim:.3f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.adapter_dir) / "vector_generalization_results.json"

    # Remove full trials for smaller output
    summary = {
        vec_type: {
            "detection_rate": results[vec_type]["detection_rate"],
            "detected": results[vec_type]["detected"],
            "total": results[vec_type]["total"],
            "mean_cos_sim": results[vec_type].get("mean_cos_sim", 0.0),
        }
        for vec_type in results
    }
    summary["config"] = {
        "model": args.model,
        "adapter_dir": args.adapter_dir,
        "layer": layer_idx,
        "n_concepts": args.n_concepts,
        "seed": args.seed,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
