"""Dagster asset for generating audit and coverage reports."""

from dagster import asset, AssetExecutionContext
from typing import Dict, Any
import json
from datetime import datetime

from src.database.surreal_client import SurrealDBSync


@asset(deps=["surreal_upsert_relevance_scores"])
def audit_report(
    context: AssetExecutionContext,
    umls_raw_load: Dict[str, Any],
    relevance_scores_calc: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate audit report showing coverage of concepts across domains.
    
    Returns:
        Dictionary with audit metrics
    """
    context.log.info("Generating audit report...")
    
    client = SurrealDBSync()
    
    try:
        client.connect()
        
        # Query concept counts
        concept_count_result = client.query("SELECT count() FROM concept GROUP ALL")
        concept_count = concept_count_result[0] if concept_count_result else 0
        
        # Query metric coverage
        metric_count_result = client.query("SELECT count() FROM metric GROUP ALL")
        metric_count = metric_count_result[0] if metric_count_result else 0
        
        # Query relevance score coverage
        score_count_result = client.query("SELECT count() FROM relevance_score GROUP ALL")
        score_count = score_count_result[0] if score_count_result else 0
        
        # Query by domain
        domain_stats = {}
        df_concept = umls_raw_load["concept"]
        scores = relevance_scores_calc["scores"]
        
        for row in df_concept.iter_rows(named=True):
            domain_id = row["domain_id"]
            if domain_id not in domain_stats:
                domain_stats[domain_id] = {
                    "concept_count": 0,
                    "scored_count": 0,
                    "high_score_count": 0,  # score > 0.85
                }
            
            domain_stats[domain_id]["concept_count"] += 1
            
            concept_id = row["concept_id"]
            if concept_id in scores:
                domain_stats[domain_id]["scored_count"] += 1
                if scores[concept_id]["score"] > 0.85:
                    domain_stats[domain_id]["high_score_count"] += 1
        
        # Build report
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_concepts": len(df_concept),
                "concepts_in_db": concept_count,
                "concepts_with_metrics": metric_count,
                "concepts_with_scores": score_count,
                "coverage_percentage": (score_count / len(df_concept) * 100) if len(df_concept) > 0 else 0,
            },
            "domain_statistics": domain_stats,
        }
        
        context.log.info(f"Report generated: {len(domain_stats)} domains analyzed")
        context.log.info(f"Coverage: {report['summary']['coverage_percentage']:.2f}%")
        
        # Save report to file
        report_path = "audit_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        context.log.info(f"Report saved to {report_path}")
        
        return report
    finally:
        client.close()

