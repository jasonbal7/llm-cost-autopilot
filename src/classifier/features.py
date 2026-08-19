import re

def extract_features(prompt: str) -> dict:
    """Extracts structural and semantic features from a prompt."""
    prompt_lower = prompt.lower()
    
    # Token Count Proxy (Word Count)
    words = re.findall(r'\b\w+\b', prompt_lower)      # prompt lower = ["I love coding"]
    word_count = len(words)                           # words = ["I", "love", "coding"]
    
    # Context Detection
    # Checks for quotes, code blocks, or explicit context phrases
    has_context = 1 if re.search(r'(```|"""|"|\'|given the following|based on the text|below:)', prompt_lower) else 0
    
    # Output Format Complexity
    format_keywords = ['json', 'csv', 'yaml', 'markdown', 'table', 'bullet points', 'list', 'xml']
    format_complexity = sum(1 for f in format_keywords if f in prompt_lower)
    
    # Constraint Counting
    constraint_keywords = ['must', 'only', 'exactly', 'limit', 'less than', 'more than', 'maximum', 'minimum', 'sentences', 'words', 'strictly']
    num_constraints = sum(1 for c in constraint_keywords if c in prompt_lower)
    
    # Tier-Specific Instruction Matching
    # Tier 1: reformatting, extraction, basic Q&A
    t1_words = {'extract', 'format', 'convert', 'what', 'who', 'where', 'when', 'find', 'remove', 'email', 'clean', 'parse', 'capital'}
    
    # Tier 2: summarization, classification, structured analysis
    t2_words = {'summarize', 'classify', 'categorize', 'group', 'analyze', 'compare', 'difference', 'draft', 'sentiment', 'bullet points', 
                'json', 'sql', 'rewrite'}
    
    # Tier 3: multi-step reasoning, creative generation, nuanced judgment calls
    t3_words = {'critique', 'reason', 'judge', 'design', 'architect', 'imagine', 'evaluate', 'synthesize', 'mitigation', 'diagnose'}
    
    return {
        #"char_length": len(prompt),            # commented out cause it caused the model to predict high every time
        #"word_count": word_count,
        "has_context": has_context,
        "format_complexity": format_complexity,
        "num_constraints": num_constraints,
        "t1_keyword_count": sum(1 for w in words if w in t1_words),
        "t2_keyword_count": sum(1 for w in words if w in t2_words),
        "t3_keyword_count": sum(1 for w in words if w in t3_words),
    }