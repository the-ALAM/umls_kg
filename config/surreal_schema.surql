-- SurrealDB Schema for UMLS Knowledge Graph System
-- Layer 1: Raw Ground Truth

-- Concept table: Primary nodes
DEFINE TABLE concept SCHEMAFULL;
DEFINE FIELD concept_id ON concept TYPE int; -- string?
DEFINE FIELD name ON concept TYPE string;
DEFINE FIELD vocabulary_id ON concept TYPE string;
DEFINE FIELD domain_id ON concept TYPE string;
DEFINE FIELD concept_code ON concept TYPE string;
DEFINE FIELD concept_class_id ON concept TYPE string;
DEFINE FIELD standard_concept ON concept TYPE string;
DEFINE FIELD valid_start_date ON concept TYPE datetime;
DEFINE FIELD valid_end_date ON concept TYPE datetime;
DEFINE FIELD invalid_reason ON concept TYPE option<string>;
DEFINE INDEX idx_concept_vocab ON concept FIELDS vocabulary_id;
DEFINE INDEX idx_concept_domain ON concept FIELDS domain_id;
DEFINE INDEX idx_concept_code ON concept FIELDS concept_code;

-- Relationship edge: relates_to
DEFINE TABLE relates_to SCHEMAFULL;
DEFINE FIELD in ON relates_to TYPE record<concept>;
DEFINE FIELD out ON relates_to TYPE record<concept>;
DEFINE FIELD relationship_id ON relates_to TYPE string;
DEFINE FIELD category ON relates_to TYPE string;
DEFINE FIELD weight ON relates_to TYPE float;
DEFINE FIELD valid_start_date ON relates_to TYPE datetime;
DEFINE FIELD valid_end_date ON relates_to TYPE datetime;
DEFINE INDEX idx_rel_type ON relates_to FIELDS relationship_id;
DEFINE INDEX idx_rel_category ON relates_to FIELDS category;

-- Ancestor edge: is_ancestor_of
DEFINE TABLE is_ancestor_of SCHEMAFULL;
DEFINE FIELD in ON is_ancestor_of TYPE record<concept>;
DEFINE FIELD out ON is_ancestor_of TYPE record<concept>;
DEFINE FIELD min_levels_of_separation ON is_ancestor_of TYPE int;
DEFINE FIELD max_levels_of_separation ON is_ancestor_of TYPE int;
DEFINE INDEX idx_ancestor_levels ON is_ancestor_of FIELDS min_levels_of_separation;

-- Vocabulary reference table
DEFINE TABLE vocabulary SCHEMAFULL;
DEFINE FIELD vocabulary_id ON vocabulary TYPE string;
DEFINE FIELD vocabulary_name ON vocabulary TYPE string;
DEFINE FIELD vocabulary_reference ON vocabulary TYPE string;
DEFINE FIELD vocabulary_version ON vocabulary TYPE string;
DEFINE FIELD vocabulary_concept_id ON vocabulary TYPE int;

-- Domain reference table
DEFINE TABLE domain SCHEMAFULL;
DEFINE FIELD domain_id ON domain TYPE string;
DEFINE FIELD domain_name ON domain TYPE string;
DEFINE FIELD domain_concept_id ON domain TYPE int;

-- Relationship reference table
DEFINE TABLE relationship SCHEMAFULL;
DEFINE FIELD relationship_id ON relationship TYPE string;
DEFINE FIELD relationship_name ON relationship TYPE string;
DEFINE FIELD is_hierarchical ON relationship TYPE int;
DEFINE FIELD defines_ancestry ON relationship TYPE int;
DEFINE FIELD reverse_relationship_id ON relationship TYPE string;
DEFINE FIELD relationship_concept_id ON relationship TYPE int;

-- Layer 2: Metric Sidecar
DEFINE TABLE metric SCHEMAFULL;
DEFINE FIELD concept_id ON metric TYPE record<concept>;
DEFINE FIELD source_authority ON metric TYPE float;
DEFINE FIELD centrality ON metric TYPE float;
DEFINE FIELD semantic_embed ON metric TYPE array<float>;
DEFINE FIELD hierarchy_depth ON metric TYPE int;
DEFINE FIELD synonym_count ON metric TYPE int;
DEFINE FIELD model_version ON metric TYPE string;
DEFINE FIELD fingerprint ON metric TYPE string;
DEFINE FIELD timestamp ON metric TYPE datetime;
DEFINE INDEX idx_metric_concept ON metric FIELDS concept_id;
DEFINE INDEX idx_metric_version ON metric FIELDS model_version;
-- Vector index for semantic embeddings (SurrealDB MTREE)
DEFINE INDEX idx_semantic_embed ON metric FIELDS semantic_embed MTREE DIMENSION 1536;

-- Layer 3: Relevance Scores (cached view/table)
DEFINE TABLE relevance_score SCHEMAFULL;
DEFINE FIELD concept_id ON relevance_score TYPE record<concept>;
DEFINE FIELD domain_cluster_id ON relevance_score TYPE string;
DEFINE FIELD score ON relevance_score TYPE float;
DEFINE FIELD s_struct ON relevance_score TYPE float;
DEFINE FIELD s_sem ON relevance_score TYPE float;
DEFINE FIELD s_density ON relevance_score TYPE float;
DEFINE FIELD s_authority ON relevance_score TYPE float;
DEFINE FIELD timestamp ON relevance_score TYPE datetime;
DEFINE INDEX idx_relevance_concept ON relevance_score FIELDS concept_id;
DEFINE INDEX idx_relevance_domain ON relevance_score FIELDS domain_cluster_id;
DEFINE INDEX idx_relevance_score ON relevance_score FIELDS score;

