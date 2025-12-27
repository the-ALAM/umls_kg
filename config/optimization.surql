-- SurrealDB Optimization Queries for UMLS Knowledge Graph

-- Vector index for semantic embeddings (already in schema, but can be recreated here)
-- DEFINE INDEX idx_semantic_embed ON metric FIELDS semantic_embed MTREE DIMENSION 1536;

-- Additional composite indexes for query optimization
DEFINE INDEX idx_relevance_domain_score ON relevance_score FIELDS domain_cluster_id, score;
DEFINE INDEX idx_metric_concept_version ON metric FIELDS concept_id, model_version;
DEFINE INDEX idx_concept_vocab_domain ON concept FIELDS vocabulary_id, domain_id;

-- Graph traversal optimization indexes
DEFINE INDEX idx_relates_to_in_out ON relates_to FIELDS in, out;
DEFINE INDEX idx_ancestor_in_out ON is_ancestor_of FIELDS in, out;

-- Query performance indexes
DEFINE INDEX idx_relevance_timestamp ON relevance_score FIELDS timestamp;
DEFINE INDEX idx_metric_timestamp ON metric FIELDS timestamp;

-- Analyze table statistics (SurrealDB doesn't have ANALYZE, but we can query for stats)
-- SELECT count() FROM concept GROUP ALL;
-- SELECT count() FROM relates_to GROUP ALL;
-- SELECT count() FROM metric GROUP ALL;
-- SELECT count() FROM relevance_score GROUP ALL;

