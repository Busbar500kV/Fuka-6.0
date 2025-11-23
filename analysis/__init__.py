"""
Fuka-6.0 analysis package.

Convenience exports for clustering, decoding, and graph analysis.
"""

from .cluster import (
    CosineClusterConfig,
    DBSCANConfig,
    cluster_cosine_incremental,
    cluster_with_fallback,
    pca_project,
)
from .decode import (
    build_supervised_mapping,
    build_unsupervised_labels,
    decode_sequence,
    decode_accuracy,
    extract_token_chain,
    join_token_chain,
    transition_counts,
)
from .transition_graph import (
    GraphConfig,
    build_transition_graph,
    simplify_graph_by_degree,
)

__all__ = [
    "CosineClusterConfig",
    "DBSCANConfig",
    "cluster_cosine_incremental",
    "cluster_with_fallback",
    "pca_project",
    "build_supervised_mapping",
    "build_unsupervised_labels",
    "decode_sequence",
    "decode_accuracy",
    "extract_token_chain",
    "join_token_chain",
    "transition_counts",
    "GraphConfig",
    "build_transition_graph",
    "simplify_graph_by_degree",
]