from typing import List, Dict

class FedUseCaseClassifier:
    def _calculate_technical_alignment(self, use_case_text: str, category_keywords: List[Dict]) -> float:
        """Calculate technical alignment score based on keyword matches.
        
        Args:
            use_case_text: The use case text to analyze
            category_keywords: List of keyword dictionaries with name and relevance_score
            
        Returns:
            float: Technical alignment score between 0 and 1
        """
        if not category_keywords:
            return 0.0
        
        use_case_text = use_case_text.lower()
        total_score = 0.0
        max_possible = 0.0
        
        for keyword in category_keywords:
            keyword_name = keyword['name'].lower()
            relevance = keyword.get('relevance_score', 0.5)
            max_possible += relevance
            
            # Check for exact match
            if keyword_name in use_case_text:
                total_score += relevance
                continue
            
            # Check for phrase match (handle multi-word keywords)
            if ' ' in keyword_name:
                words = keyword_name.split()
                if all(word in use_case_text for word in words):
                    total_score += relevance * 0.8  # Slightly lower score for non-exact phrase match
            
        return min(1.0, total_score / max_possible if max_possible > 0 else 0.0)

    def _calculate_business_alignment(self, use_case_text: str, business_terms: List[Dict]) -> float:
        """Calculate business alignment score based on business term matches.
        
        Args:
            use_case_text: The use case text to analyze
            business_terms: List of business term dictionaries with name and relevance_score
            
        Returns:
            float: Business alignment score between 0 and 1
        """
        if not business_terms:
            return 0.0
        
        use_case_text = use_case_text.lower()
        total_score = 0.0
        max_possible = 0.0
        
        for term in business_terms:
            term_name = term['name'].lower()
            relevance = term.get('relevance_score', 0.5)
            max_possible += relevance
            
            # Check for exact match
            if term_name in use_case_text:
                total_score += relevance
                continue
            
            # Check for phrase match
            if ' ' in term_name:
                words = term_name.split()
                if all(word in use_case_text for word in words):
                    total_score += relevance * 0.8
            
        return min(1.0, total_score / max_possible if max_possible > 0 else 0.0) 