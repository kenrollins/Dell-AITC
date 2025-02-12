from .services.classifier import Classifier
from .services.neo4j_classifier import Neo4jClassifier

# Singleton instances for services
classifier = Classifier()
neo4j_classifier = Neo4jClassifier(
    uri="bolt://localhost:7687",  # Default local Neo4j
    username="neo4j",            # Default Neo4j username
    password="password"          # Default Neo4j password
)

def get_classifier() -> Classifier:
    """Get the singleton Classifier instance"""
    return classifier

def get_neo4j_classifier() -> Neo4jClassifier:
    """Get the singleton Neo4jClassifier instance"""
    return neo4j_classifier
