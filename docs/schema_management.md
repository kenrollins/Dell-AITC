# Schema Management Tools

## Overview
This document describes the schema management tools available for the Dell-AITC project. These tools help maintain schema integrity, track changes, and monitor performance.

## Tools

### 1. Schema Validator
Located in `scripts/database/maintenance/schema_validator.py`

#### Features
- Validates required node properties
- Checks relationship definitions
- Enforces keyword type constraints
- Validates version metadata
- Provides detailed error reporting

#### Usage
```bash
python schema_validator.py path/to/schema.json
```

#### Validation Rules
- All nodes must have required properties (id, name, status, etc.)
- Keywords must use approved types:
  - technical_keywords
  - capabilities
  - business_language
- All nodes must include version metadata
- Relationships must define source, target, and properties

### 2. Schema Monitor
Located in `scripts/database/maintenance/schema_monitor.py`

#### Features
- Tracks node and relationship counts
- Monitors property usage
- Measures query performance
- Generates CSV reports
- Provides human-readable summaries

#### Usage
```bash
python schema_monitor.py --metrics-dir ./metrics
```

#### Metrics Collected
1. Node Counts
   - Count per node label
   - Historical tracking

2. Relationship Counts
   - Count per relationship type
   - Usage patterns

3. Property Usage
   - Property distribution across node types
   - Frequency analysis

4. Query Performance
   - Common query execution times
   - Performance trending

#### Configuration
The schema monitor uses environment variables for database connection:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

## Best Practices

### Schema Updates
1. Run schema validation before and after changes
2. Collect metrics before major updates
3. Monitor performance impact
4. Document all schema changes

### Performance Monitoring
1. Regular metric collection (daily/weekly)
2. Track trends over time
3. Set up alerts for significant changes
4. Review query performance regularly

### Data Quality
1. Validate all new data against schema
2. Monitor property usage patterns
3. Check for orphaned nodes/relationships
4. Maintain data consistency

## Integration

### With CI/CD
```yaml
schema-validation:
  script:
    - python schema_validator.py schema.json
    - python schema_monitor.py --metrics-dir ./metrics
  artifacts:
    paths:
      - metrics/
```

### With Development Workflow
1. Pre-commit hooks for schema validation
2. Automated metric collection
3. Performance impact assessment
4. Documentation updates

## Troubleshooting

### Common Issues
1. Schema Validation Failures
   - Check error messages
   - Verify property requirements
   - Validate relationship definitions

2. Performance Issues
   - Review query metrics
   - Check node/relationship counts
   - Analyze property usage

3. Data Inconsistencies
   - Run validation scripts
   - Check error logs
   - Review metric reports

## Future Enhancements
1. Real-time monitoring
2. Automated schema optimization
3. Advanced performance analytics
4. Integration with monitoring systems 