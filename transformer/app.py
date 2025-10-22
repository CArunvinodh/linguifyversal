import ssl
import random
import warnings
import os

import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Disable unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Global NLP model with memory optimization
try:
    NLP_GLOBAL = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
except OSError:
    # Fallback for environments without spacy model
    NLP_GLOBAL = None
    print("⚠️ spaCy model not available, using fallback mode")

def download_nltk_resources():
    """
    Download required NLTK resources if not already installed.
    """
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    resources = ['punkt', 'averaged_perceptron_tagger', 'wordnet']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")


class AcademicTextHumanizer:
    """
    Lightweight text humanizer optimized for serverless environments.
    """

    def __init__(
        self,
        p_passive=0.2,
        p_synonym_replacement=0.3,
        p_academic_transition=0.3,
        seed=None
    ):
        if seed is not None:
            random.seed(seed)

        # Load spaCy with minimal components for memory efficiency
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
        except OSError:
            self.nlp = None
            print("⚠️ spaCy model not available - using basic text processing")

        # Use lightweight approach instead of heavy sentence transformers
        self.model = None  # Skip sentence-transformers to save memory
        
        # Conservative probabilities for serverless
        self.p_passive = min(p_passive, 0.3)  # Cap at 30%
        self.p_synonym_replacement = min(p_synonym_replacement, 0.3)
        self.p_academic_transition = min(p_academic_transition, 0.3)

        # Common academic transitions
        self.academic_transitions = [
            "Moreover,", "Additionally,", "Furthermore,", "Hence,", 
            "Therefore,", "Consequently,", "Nonetheless,", "Nevertheless,"
        ]

    def humanize_text(self, text, use_passive=False, use_synonyms=False):
        """Humanize text with memory safety limits"""
        
        # Input validation
        if not text or not text.strip():
            return text
            
        # Strict length limits for serverless
        if len(text) > 10000:  # 10K character limit
            return "Error: Input text too long for processing"
        
        try:
            # Use basic sentence splitting if spaCy fails
            if self.nlp is None:
                sentences = [s.strip() for s in text.split('.') if s.strip()]
            else:
                doc = self.nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents]

            transformed_sentences = []
            
            for sentence_str in sentences:
                if not sentence_str:
                    continue
                    
                # 1. Expand contractions
                sentence_str = self.expand_contractions(sentence_str)

                # 2. Possibly add academic transitions (with lower probability)
                if random.random() < self.p_academic_transition:
                    sentence_str = self.add_academic_transitions(sentence_str)

                # 3. Optionally convert to passive (simplified)
                if use_passive and random.random() < self.p_passive:
                    sentence_str = self.convert_to_passive_simple(sentence_str)

                # 4. Optionally replace words with synonyms (simplified)
                if use_synonyms and random.random() < self.p_synonym_replacement:
                    sentence_str = self.replace_with_synonyms_simple(sentence_str)

                transformed_sentences.append(sentence_str)

            return ' '.join(transformed_sentences)
            
        except Exception as e:
            return f"Error processing text: {str(e)}"

    def expand_contractions(self, sentence):
        """Expand common contractions"""
        contractions = {
            "n't": " not", "'re": " are", "'s": " is", "'ll": " will",
            "'ve": " have", "'d": " would", "'m": " am",
            "can't": "cannot", "won't": "will not", "don't": "do not",
            "doesn't": "does not", "isn't": "is not", "aren't": "are not"
        }
        
        # Simple replacement without tokenization to save memory
        result = sentence
        for contraction, expansion in contractions.items():
            result = result.replace(contraction, expansion)
        return result

    def add_academic_transitions(self, sentence):
        """Add academic transition words"""
        transition = random.choice(self.academic_transitions)
        return f"{transition} {sentence}"

    def convert_to_passive_simple(self, sentence):
        """Simplified passive voice conversion"""
        words = sentence.split()
        if len(words) >= 3:
            # Very basic pattern matching for common structures
            if words[1] in ['is', 'are', 'was', 'were']:
                return sentence  # Already passive-like
                
            # Simple subject-verb-object to object-verb-by-subject
            if len(words) >= 4:
                return f"{words[2]} {words[1]} by {words[0]} { ' '.join(words[3:])}"
        
        return sentence

    def replace_with_synonyms_simple(self, sentence):
        """Simplified synonym replacement without heavy models"""
        words = sentence.split()
        new_words = []
        
        for word in words:
            # Only process words longer than 3 characters to avoid common words
            if len(word) > 3 and word.isalpha() and random.random() < 0.3:
                synonyms = self._get_simple_synonyms(word)
                if synonyms:
                    new_words.append(random.choice(synonyms))
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
                
        return ' '.join(new_words)

    def _get_simple_synonyms(self, word):
        """Get synonyms without heavy processing"""
        try:
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    lemma_name = lemma.name().replace('_', ' ')
                    if (lemma_name.lower() != word.lower() and 
                        len(lemma_name.split()) == 1 and  # Single word only
                        lemma_name.isalpha()):
                        synonyms.add(lemma_name)
                        
                        # Limit to 5 synonyms to save memory
                        if len(synonyms) >= 5:
                            break
                if len(synonyms) >= 5:
                    break
                    
            return list(synonyms) if synonyms else None
            
        except Exception:
            return None

    # Original methods kept for reference but simplified versions used above
    def convert_to_passive(self, sentence):
        """Original passive conversion (fallback)"""
        if self.nlp is None:
            return self.convert_to_passive_simple(sentence)
            
        try:
            doc = self.nlp(sentence)
            subj_tokens = [t for t in doc if t.dep_ == 'nsubj' and t.head.dep_ == 'ROOT']
            dobj_tokens = [t for t in doc if t.dep_ == 'dobj']

            if subj_tokens and dobj_tokens:
                subject = subj_tokens[0]
                dobj = dobj_tokens[0]
                verb = subject.head
                if subject.i < verb.i < dobj.i:
                    passive_str = f"{dobj.text} {verb.lemma_} by {subject.text}"
                    original_str = ' '.join(token.text for token in doc)
                    chunk = f"{subject.text} {verb.text} {dobj.text}"
                    if chunk in original_str:
                        sentence = original_str.replace(chunk, passive_str)
            return sentence
        except Exception:
            return self.convert_to_passive_simple(sentence)

    def replace_with_synonyms(self, sentence):
        """Original synonym replacement (fallback)"""
        try:
            tokens = word_tokenize(sentence)
            pos_tags = nltk.pos_tag(tokens)

            new_tokens = []
            for (word, pos) in pos_tags:
                if (pos.startswith(('J', 'N', 'V', 'R')) and 
                    len(word) > 3 and 
                    wordnet.synsets(word)):
                    if random.random() < 0.5:
                        synonyms = self._get_synonyms(word, pos)
                        if synonyms:
                            # Use random selection instead of model-based for memory
                            best_synonym = random.choice(synonyms)
                            new_tokens.append(best_synonym if best_synonym else word)
                        else:
                            new_tokens.append(word)
                    else:
                        new_tokens.append(word)
                else:
                    new_tokens.append(word)

            return ' '.join(new_tokens)
        except Exception:
            return self.replace_with_synonyms_simple(sentence)

    def _get_synonyms(self, word, pos):
        """Get synonyms with POS filtering"""
        wn_pos = None
        if pos.startswith('J'):
            wn_pos = wordnet.ADJ
        elif pos.startswith('N'):
            wn_pos = wordnet.NOUN
        elif pos.startswith('R'):
            wn_pos = wordnet.ADV
        elif pos.startswith('V'):
            wn_pos = wordnet.VERB

        synonyms = set()
        try:
            for syn in wordnet.synsets(word, pos=wn_pos):
                for lemma in syn.lemmas():
                    lemma_name = lemma.name().replace('_', ' ')
                    if lemma_name.lower() != word.lower():
                        synonyms.add(lemma_name)
            return list(synonyms)
        except Exception:
            return None
