// Drop existing constraints and indexes
CALL apoc.schema.assert({},{});

// AICategory Constraints
CREATE CONSTRAINT aicategory_id IF NOT EXISTS FOR (n:AICategory) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT aicategory_name IF NOT EXISTS FOR (n:AICategory) REQUIRE n.name IS UNIQUE;

// Agency Constraints
CREATE CONSTRAINT agency_id IF NOT EXISTS FOR (n:Agency) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT agency_name IF NOT EXISTS FOR (n:Agency) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT agency_abbr IF NOT EXISTS FOR (n:Agency) REQUIRE n.abbreviation IS UNIQUE;

// Bureau Constraints
CREATE CONSTRAINT bureau_id IF NOT EXISTS FOR (n:Bureau) REQUIRE n.id IS UNIQUE;

// UseCase Constraints
CREATE CONSTRAINT usecase_id IF NOT EXISTS FOR (n:UseCase) REQUIRE n.id IS UNIQUE;

// Keyword Constraints
CREATE CONSTRAINT keyword_id IF NOT EXISTS FOR (n:Keyword) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT keyword_name IF NOT EXISTS FOR (n:Keyword) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT keyword_type_values IF NOT EXISTS FOR (n:Keyword) 
REQUIRE n.type IN ['technical_keywords', 'capabilities', 'business_language'];

// Indexes for AICategory
CREATE INDEX aicategory_status IF NOT EXISTS FOR (n:AICategory) ON (n.status);
CREATE INDEX aicategory_maturity IF NOT EXISTS FOR (n:AICategory) ON (n.maturity_level);

// Indexes for UseCase
CREATE INDEX usecase_name IF NOT EXISTS FOR (n:UseCase) ON (n.name);
CREATE INDEX usecase_topic IF NOT EXISTS FOR (n:UseCase) ON (n.topic_area);
CREATE INDEX usecase_stage IF NOT EXISTS FOR (n:UseCase) ON (n.stage);
CREATE INDEX usecase_impact IF NOT EXISTS FOR (n:UseCase) ON (n.impact_type);
CREATE INDEX usecase_pii IF NOT EXISTS FOR (n:UseCase) ON (n.contains_pii);
CREATE INDEX usecase_ato IF NOT EXISTS FOR (n:UseCase) ON (n.has_ato);

// Indexes for Bureau
CREATE INDEX bureau_name IF NOT EXISTS FOR (n:Bureau) ON (n.name);
CREATE INDEX bureau_agency IF NOT EXISTS FOR (n:Bureau) ON (n.agency_id);

// Indexes for Keyword
CREATE INDEX keyword_type IF NOT EXISTS FOR (n:Keyword) ON (n.type);
CREATE INDEX keyword_status IF NOT EXISTS FOR (n:Keyword) ON (n.status);
CREATE INDEX keyword_relevance IF NOT EXISTS FOR (n:Keyword) ON (n.relevance_score);

// Fulltext indexes for text search
CREATE FULLTEXT INDEX usecase_text IF NOT EXISTS FOR (n:UseCase) ON EACH [n.purpose_benefits, n.outputs];
CREATE FULLTEXT INDEX aicategory_text IF NOT EXISTS FOR (n:AICategory) ON EACH [n.category_definition]; 