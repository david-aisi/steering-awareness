"""Steering vector computation using various methods (CAA, PCA, SVM)."""

import os
import gc
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from tqdm import tqdm


class VectorManager:
    """
    Manages computation and storage of steering vectors.

    Supports multiple extraction methods:
    - CAA (Contrastive Activation Addition): Mean difference method
    - PCA: Principal component of concept activations
    - SVM: Linear probe separating concept from baseline

    Args:
        model: The transformer model
        tokenizer: The tokenizer
        layer_idx: Layer to extract activations from
        device: Device for computations
    """

    PROMPT_TEMPLATE = "Human: Tell me about {}.\n\nAssistant:"

    CLOUD_TEMPLATES = [
        "Human: Tell me about {}.\n\nAssistant:",
        "Human: What is {}?\n\nAssistant:",
        "Human: Define {}.\n\nAssistant:",
        "Human: Describe the concept of {}.\n\nAssistant:",
        "Human: Explain {} to me.\n\nAssistant:",
        "Human: Give me a sentence using the word {}.\n\nAssistant:",
        "Human: How is {} used in daily life?\n\nAssistant:",
        "Human: Write a short story involving {}.\n\nAssistant:",
        "Human: What are the key characteristics of {}?\n\nAssistant:",
        "Human: What words are related to {}?\n\nAssistant:",
        "Human: Describe {} like I'm five years old.\n\nAssistant:",
        "Human: What is the opposite of {}?\n\nAssistant:",
        "Human: Why is {} important?\n\nAssistant:",
        "Human: Discuss the nature of {}.\n\nAssistant:",
        "Human: What does {} imply?\n\nAssistant:",
        "Human: Imagine a world without {}.\n\nAssistant:",
        "Human: Is {} considered good or bad?\n\nAssistant:",
    ]

    def __init__(self, model, tokenizer, layer_idx: int, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.device = device
        self._baseline_mean: Optional[torch.Tensor] = None
        self._baseline_acts: Optional[np.ndarray] = None

    def _get_activation(self, text: str) -> torch.Tensor:
        """Get the activation at the last token position for the target layer."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[self.layer_idx][0, -1, :].detach().cpu()

    def compute_baseline(self, baseline_words: List[str]) -> torch.Tensor:
        """
        Compute the mean activation over baseline words.

        Args:
            baseline_words: List of neutral/common words

        Returns:
            Mean activation tensor
        """
        print(f"Computing baseline from {len(baseline_words)} words...")
        activations = []

        for word in tqdm(baseline_words, desc="Baseline"):
            text = self.PROMPT_TEMPLATE.format(word)
            act = self._get_activation(text)
            activations.append(act)

        baseline_stack = torch.stack(activations)
        self._baseline_mean = baseline_stack.mean(dim=0)
        self._baseline_acts = baseline_stack.numpy()

        return self._baseline_mean

    def compute_vectors_caa(
        self,
        concepts: List[str],
        baseline_mean: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute steering vectors using Contrastive Activation Addition.

        The CAA method computes: vector = concept_activation - baseline_mean

        Args:
            concepts: List of concept words/phrases
            baseline_mean: Precomputed baseline mean (uses cached if None)

        Returns:
            Dict mapping concept names to steering vectors
        """
        if baseline_mean is None:
            baseline_mean = self._baseline_mean
        assert baseline_mean is not None, "Must compute baseline first"

        vectors = {}
        print(f"Computing CAA vectors for {len(concepts)} concepts...")

        for concept in tqdm(concepts, desc="CAA Vectors"):
            text = self.PROMPT_TEMPLATE.format(concept)
            act = self._get_activation(text)
            vectors[concept] = act - baseline_mean

        return vectors

    def get_activation_cloud(
        self,
        concept: str,
        is_baseline: bool = False,
        baseline_words: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Generate a cloud of activations using varied prompts.

        This is necessary for PCA/SVM methods which need variance.

        Args:
            concept: The concept to get activations for
            is_baseline: If True, iterate over baseline_words instead
            baseline_words: List of baseline words (required if is_baseline=True)

        Returns:
            Array of activations [n_samples, hidden_dim]
        """
        activations = []

        if is_baseline:
            assert baseline_words is not None, "baseline_words required for baseline cloud"
            subset = baseline_words[:20]
            for word in subset:
                text = self.PROMPT_TEMPLATE.format(word)
                act = self._get_activation(text)
                activations.append(act.numpy())
        else:
            for template in self.CLOUD_TEMPLATES:
                text = template.format(concept)
                act = self._get_activation(text)
                activations.append(act.numpy())

        return np.array(activations)

    def compute_vector_pca(
        self,
        concept: str,
        baseline_mean: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Compute steering vector using PCA.

        Finds the direction of maximum variance between concept and baseline.

        Args:
            concept: The concept to compute vector for
            baseline_mean: Precomputed baseline mean as numpy array

        Returns:
            Steering vector as tensor
        """
        if baseline_mean is None:
            baseline_mean = self._baseline_mean.numpy()

        concept_acts = self.get_activation_cloud(concept)
        centered_data = concept_acts - baseline_mean

        pca = PCA(n_components=1)
        pca.fit(centered_data)

        direction = pca.components_[0]
        return torch.tensor(direction, dtype=torch.float16).to(self.device)

    def compute_vector_svm(
        self,
        concept: str,
        baseline_acts: Optional[np.ndarray] = None,
        baseline_words: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compute steering vector using SVM linear probe.

        Finds the vector orthogonal to the boundary separating concept from baseline.

        Args:
            concept: The concept to compute vector for
            baseline_acts: Precomputed baseline activations
            baseline_words: Baseline words (needed if baseline_acts not cached)

        Returns:
            Steering vector as tensor
        """
        if baseline_acts is None:
            if self._baseline_acts is not None:
                baseline_acts = self._baseline_acts
            else:
                assert baseline_words is not None, "Need baseline_words or cached baseline"
                baseline_acts = self.get_activation_cloud(
                    "", is_baseline=True, baseline_words=baseline_words
                )

        concept_acts = self.get_activation_cloud(concept)

        # Create binary classification dataset
        X = np.concatenate([concept_acts, baseline_acts])
        y = np.concatenate([np.ones(len(concept_acts)), np.zeros(len(baseline_acts))])

        # Train linear SVM
        clf = LinearSVC(dual="auto", max_iter=1000)
        clf.fit(X, y)

        # Normal vector to the hyperplane
        direction = clf.coef_[0]
        direction = direction / np.linalg.norm(direction)

        return torch.tensor(direction, dtype=torch.float16).to(self.device)

    def save_vectors(self, vectors: Dict[str, torch.Tensor], path: str):
        """Save vectors to disk."""
        torch.save(vectors, path)
        print(f"Saved {len(vectors)} vectors to {path}")

    def load_vectors(self, path: str) -> Dict[str, torch.Tensor]:
        """Load vectors from disk."""
        vectors = torch.load(path)
        print(f"Loaded {len(vectors)} vectors from {path}")
        return vectors


def quantify_ood_metrics(
    vectors_dict: Dict[str, torch.Tensor],
    train_concepts: List[str],
    eval_suites: Dict[str, List[str]],
) -> Dict[str, Dict[str, float]]:
    """
    Quantify the distance between training distribution and OOD test suites.

    Computes:
    - Mean cosine distance
    - Silhouette score (cluster separation)
    - Maximum Mean Discrepancy (distribution difference)

    Args:
        vectors_dict: Dict of concept -> vector
        train_concepts: List of training concept names
        eval_suites: Dict of suite_name -> list of concepts

    Returns:
        Dict of suite_name -> metrics dict
    """
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel

    train_vecs = torch.stack(
        [vectors_dict[c] for c in train_concepts if c in vectors_dict]
    ).numpy()

    results = {}

    for suite_name, test_concepts in eval_suites.items():
        valid_test = [c for c in test_concepts if c in vectors_dict]
        if not valid_test:
            continue

        test_vecs = torch.stack([vectors_dict[c] for c in valid_test]).numpy()

        # Mean Cosine Distance
        cos_sim = cosine_similarity(test_vecs, train_vecs)
        avg_cos_dist = 1 - np.mean(cos_sim)

        # Silhouette Score
        combined_data = np.vstack([train_vecs, test_vecs])
        labels = np.array([0] * len(train_vecs) + [1] * len(test_vecs))
        sil = silhouette_score(combined_data, labels)

        # Maximum Mean Discrepancy
        XX = rbf_kernel(train_vecs, train_vecs)
        YY = rbf_kernel(test_vecs, test_vecs)
        XY = rbf_kernel(train_vecs, test_vecs)
        mmd = XX.mean() + YY.mean() - 2 * XY.mean()

        results[suite_name] = {
            "cosine_distance": round(float(avg_cos_dist), 4),
            "silhouette_score": round(float(sil), 4),
            "mmd": round(float(mmd), 4),
            "n_samples": len(valid_test),
        }

    return results


def load_vectors(path: str) -> Dict[str, torch.Tensor]:
    """Load vectors from a .pt file."""
    return torch.load(path)
