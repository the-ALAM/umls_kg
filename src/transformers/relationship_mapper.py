"""Map relationship_id to categories and weights (W_rel mapping)."""

from typing import Dict, Tuple


# Relationship weight mapping (W_rel)
RELATIONSHIP_WEIGHTS: Dict[str, float] = {
    # Equivalence (1.0)
    "mapped_from": 1.0,
    "concept_same_as": 1.0,
    "maps_to": 1.0,
    
    # Structural (0.8)
    "is_a": 0.8,
    "subsumes": 0.8,
    "isa": 0.8,
    
    # Clinical (0.5)
    "has_finding_site": 0.5,
    "due_to": 0.5,
    "caused_by": 0.5,
    "associated_with": 0.5,
    "has_dose_form": 0.5,
    "has_ingredient": 0.5,
    
    # Meta (0.3)
    "mapped_to": 0.3,
    "concept_replaced_by": 0.3,
    "replaced_by": 0.3,
}

# Relationship category mapping
#TODO - should i change it to a Dict[str, List[str]]
RELATIONSHIP_CATEGORIES: Dict[str, str] = {
    # Equivalence
    "mapped_from": "equivalence",
    "concept_same_as": "equivalence",
    "maps_to": "equivalence",
    
    # Structural
    "is_a": "structural",
    "subsumes": "structural",
    "isa": "structural",
    
    # Clinical
    "has_finding_site": "clinical",
    "due_to": "clinical",
    "caused_by": "clinical",
    "associated_with": "clinical",
    "has_dose_form": "clinical",
    "has_ingredient": "clinical",
    
    # Meta
    "mapped_to": "meta",
    "concept_replaced_by": "meta",
    "replaced_by": "meta",
}


def get_relationship_weight(relationship_id: str) -> float:
    """
    Get weight for a relationship_id.
    
    Args:
        relationship_id: Relationship identifier
        
    Returns:
        Weight value (default: 0.3 for unknown relationships)
    """
    # Normalize relationship_id (lowercase, strip)
    normalized = relationship_id.lower().strip()
    
    # Direct lookup
    if normalized in RELATIONSHIP_WEIGHTS:
        return RELATIONSHIP_WEIGHTS[normalized]
    
    # Pattern matching for common variations
    if "map" in normalized and "from" in normalized:
        return 1.0
    if "same" in normalized or "equivalent" in normalized:
        return 1.0
    if "is_a" in normalized or "isa" in normalized:
        return 0.8
    if "subsum" in normalized or "parent" in normalized:
        return 0.8
    if "finding" in normalized or "site" in normalized:
        return 0.5
    if "due" in normalized or "cause" in normalized:
        return 0.5
    if "replace" in normalized:
        return 0.3
    
    # Default for unknown relationships
    return 0.3


def get_relationship_category(relationship_id: str) -> str:
    """
    Get category for a relationship_id.
    
    Args:
        relationship_id: Relationship identifier
        
    Returns:
        Category string (default: "meta" for unknown relationships)
    """
    # Normalize relationship_id
    normalized = relationship_id.lower().strip()
    
    # Direct lookup
    if normalized in RELATIONSHIP_CATEGORIES:
        return RELATIONSHIP_CATEGORIES[normalized]
    
    # Pattern matching
    if "map" in normalized and "from" in normalized:
        return "equivalence"
    if "same" in normalized or "equivalent" in normalized:
        return "equivalence"
    if "is_a" in normalized or "isa" in normalized:
        return "structural"
    if "subsum" in normalized or "parent" in normalized:
        return "structural"
    if "finding" in normalized or "site" in normalized or "due" in normalized or "cause" in normalized:
        return "clinical"
    if "replace" in normalized:
        return "meta"
    
    # Default
    return "meta"


def get_relationship_info(relationship_id: str) -> Tuple[str, float]:
    """
    Get both category and weight for a relationship_id.
    
    Args:
        relationship_id: Relationship identifier
        
    Returns:
        Tuple of (category, weight)
    """
    category = get_relationship_category(relationship_id)
    weight = get_relationship_weight(relationship_id)
    return category, weight


def is_structural_relationship(relationship_id: str) -> bool:
    """
    Check if relationship is structural (hierarchical).
    
    Args:
        relationship_id: Relationship identifier
        
    Returns:
        True if structural, False otherwise
    """
    category = get_relationship_category(relationship_id)
    return category == "structural"


def is_equivalence_relationship(relationship_id: str) -> bool:
    """
    Check if relationship is equivalence.
    
    Args:
        relationship_id: Relationship identifier
        
    Returns:
        True if equivalence, False otherwise
    """
    category = get_relationship_category(relationship_id)
    return category == "equivalence"

