def _validate_schema_compliance(self):
    """Validate that all nodes have required properties according to schema."""
    query = """
        MATCH (u:UseCase)
        WHERE u.id IS NULL
            OR u.name IS NULL
            OR u.topic_area IS NULL
            OR u.stage IS NULL
            OR u.impact_type IS NULL
            OR u.created_at IS NULL
            OR u.last_updated IS NULL
        RETURN count(u) as invalid_nodes
    """
    result = self.db.run(query).single()
    if result and result["invalid_nodes"] > 0:
        logger.warning(f"Found {result['invalid_nodes']} use cases with missing required properties")
        return False
    return True 