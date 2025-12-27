"""Vocabulary authority mapping (W_vocab)."""

from typing import Dict


# Vocabulary authority weights (W_vocab)
VOCABULARY_AUTHORITY: Dict[str, float] = {
    "SNOMED": 1.0,  # Clinical gold standard
    "ICD10CM": 0.9,  # Standard for billing and DRG mapping
    "ICD10": 0.9,
    "LOINC": 0.8,  # Specificity for lab/observation data
    "RxNorm": 0.8,  # Standard for pharmacological entities
    "ICD9CM": 0.7,
    "CPT4": 0.7,
    "HCPCS": 0.7,
    "ATC": 0.7,
    "NDC": 0.6,
    "UCUM": 0.6,
    "OMOP Extension": 0.5,
    "OMOP Genomic": 0.5,
}


def get_vocabulary_authority(vocabulary_id: str) -> float:
    """
    Get authority score for a vocabulary_id.
    
    Args:
        vocabulary_id: Vocabulary identifier
        
    Returns:
        Authority score (default: 0.5 for unknown vocabularies)
    """
    # Direct lookup
    if vocabulary_id in VOCABULARY_AUTHORITY:
        return VOCABULARY_AUTHORITY[vocabulary_id]
    
    # Pattern matching
    vocabulary_upper = vocabulary_id.upper()
    if "SNOMED" in vocabulary_upper:
        return 1.0
    if "ICD10" in vocabulary_upper:
        return 0.9
    if "LOINC" in vocabulary_upper:
        return 0.8
    if "RXNORM" in vocabulary_upper or "RX" in vocabulary_upper:
        return 0.8
    if "ICD9" in vocabulary_upper:
        return 0.7
    if "CPT" in vocabulary_upper or "HCPCS" in vocabulary_upper:
        return 0.7
    
    # Default for unknown vocabularies
    return 0.5

