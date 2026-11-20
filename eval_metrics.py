"""Graph metrics and evaluation pipeline for multilingual news verification."""

from typing import Dict, Any, List
import networkx as nx


def compute_credibility_score(graph_features: List[float], text_length: int) -> float:
    """Compute normalized credibility score based on graph density and entity counts.
    
    Raises:
        ValueError: If graph_features is empty or text_length is non-positive.
    """
    if not graph_features or len(graph_features) < 3:
        raise ValueError("graph_features must contain [nodes, edges, density]")
    if text_length <= 0:
        raise ValueError("text_length must be a positive integer")

    nodes, edges, density = graph_features[0], graph_features[1], graph_features[2]
    entity_ratio = nodes / max(1, text_length)
    score = (density * 0.5) + (min(1.0, entity_ratio) * 0.5)
    return round(float(score), 4)


def batch_evaluate_claims(claims: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate batch of multilingual claims safely with error boundary handling."""
    results = []
    errors = 0

    for claim in claims:
        try:
            text = claim.get("text", "")
            features = claim.get("features", [0, 0, 0.0])
            score = compute_credibility_score(features, len(text.split()))
            results.append({"id": claim.get("id"), "score": score, "valid": True})
        except Exception as err:
            errors += 1
            results.append({"id": claim.get("id"), "score": 0.0, "valid": False, "error": str(err)})

    return {
        "total": len(claims),
        "evaluated": len(results),
        "errors": errors,
        "results": results,
    }
