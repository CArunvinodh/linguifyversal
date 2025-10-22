import ssl
import random
import warnings
import os

import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet

# Disable unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Global NLP model with memory optimization
try:
    # Load spaCy with sentencizer for sentence boundary detection
    NLP_GLOBAL = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    # Add sentencizer to handle sentence boundaries
    if "sentencizer" not in NLP_GLOBAL.pipe_names:
        NLP_GLOBAL.add_pipe("sentencizer")
    print("✅ spaCy model loaded successfully with sentencizer")
except OSError:
    # Fallback for environments without spacy model
    NLP_GLOBAL = None
    print("⚠️ spaCy model not available, using fallback mode")

def download_nltk_resources():
    """
    Download required NLTK resources if not already installed.
    Optimized for serverless environments.
    """
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Minimal required resources
    resources = ['punkt', 'averaged_perceptron_tagger', 'wordnet']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"⚠️ Error downloading {resource}: {str(e)}")

class AcademicTextHumanizer:
    """
    Lightweight text humanizer optimized for serverless environments.
    No heavy dependencies like sentence-transformers.
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

        # Load spaCy with sentencizer for sentence boundary detection
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
            # Add sentencizer to handle sentence boundaries
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")
            print("✅ AcademicTextHumanizer: spaCy loaded with sentencizer")
        except OSError:
            self.nlp = None
            print("⚠️ AcademicTextHumanizer: spaCy not available - using basic text processing")

        # Conservative probabilities for serverless
        self.p_passive = min(p_passive, 0.3)  # Cap at 30%
        self.p_synonym_replacement = min(p_synonym_replacement, 0.3)
        self.p_academic_transition = min(p_academic_transition, 0.3)

        # Common academic transitions
        self.academic_transitions = [
            "Moreover,", "Additionally,", "Furthermore,", "Hence,", 
            "Therefore,", "Consequently,", "Nonetheless,", "Nevertheless,"
        ]

        # Common contractions mapping
        self.contractions_map = {
            "n't": " not", "'re": " are", "'s": " is", "'ll": " will",
            "'ve": " have", "'d": " would", "'m": " am",
            "can't": "cannot", "won't": "will not", "don't": "do not",
            "doesn't": "does not", "isn't": "is not", "aren't": "are not",
            "wasn't": "was not", "weren't": "were not", "haven't": "have not",
            "hasn't": "has not", "hadn't": "had not", "wouldn't": "would not",
            "shouldn't": "should not", "couldn't": "could not", "mightn't": "might not"
        }

    def humanize_text(self, text, use_passive=False, use_synonyms=False):
        """
        Humanize text with memory safety limits and strict input validation.
        """
        # Input validation
        if not text or not isinstance(text, str):
            return "Error: Invalid input text"
            
        text = text.strip()
        if not text:
            return "Please enter some text to process."

        # Strict length limits for serverless
        if len(text) > 10000:  # 10K character limit
            return "Error: Input text too long for processing (max 10,000 characters)"
        
        try:
            # Get sentences using the appropriate method
            sentences = self._get_sentences(text)
            
            if not sentences:
                return "No valid sentences found to process."

            transformed_sentences = []
            
            for sentence in sentences:
                if not sentence:
                    continue
                    
                current_sentence = sentence

                # 1. Expand contractions
                current_sentence = self.expand_contractions(current_sentence)

                # 2. Possibly add academic transitions (with lower probability)
                if random.random() < self.p_academic_transition:
                    current_sentence = self.add_academic_transitions(current_sentence)

                # 3. Optionally convert to passive (simplified)
                if use_passive and random.random() < self.p_passive:
                    current_sentence = self.convert_to_passive_simple(current_sentence)

                # 4. Optionally replace words with synonyms (simplified)
                if use_synonyms and random.random() < self.p_synonym_replacement:
                    current_sentence = self.replace_with_synonyms_simple(current_sentence)

                transformed_sentences.append(current_sentence)

            result = ' '.join(transformed_sentences)
            
            # Final safety check
            if len(result) > 15000:  # Slightly larger to account for expansions
                return "Error: Output too large after processing"
                
            return result
            
        except Exception as e:
            return f"Error processing text: {str(e)}"

    def _get_sentences(self, text):
        """
        Extract sentences using the best available method.
        """
        # Use NLTK sent_tokenize as primary method (most reliable)
        try:
            sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
            if sentences:
                return sentences
        except Exception as e:
            print(f"⚠️ NLTK sent_tokenize failed: {e}")
        
        # Fallback to spaCy with sentencizer if available
        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                if sentences:
                    return sentences
            except Exception as e:
                print(f"⚠️ spaCy sentence segmentation failed: {e}")
        
        # Final fallback: simple period-based splitting
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        return sentences

    def expand_contractions(self, sentence):
        """
        Expand common contractions efficiently.
        """
        if not sentence:
            return sentence
            
        result = sentence
        # Sort by length descending to handle longer contractions first
        contractions_sorted = sorted(self.contractions_map.items(), 
                                   key=lambda x: len(x[0]), 
                                   reverse=True)
        
        for contraction, expansion in contractions_sorted:
            result = result.replace(contraction, expansion)
            
        return result

    def add_academic_transitions(self, sentence):
        """
        Add academic transition words at the beginning of sentences.
        """
        if not sentence:
            return sentence
            
        transition = random.choice(self.academic_transitions)
        return f"{transition} {sentence}"

    def convert_to_passive_simple(self, sentence):
        """
        Simplified passive voice conversion using basic pattern matching.
        """
        if not sentence or len(sentence.split()) < 3:
            return sentence

        words = sentence.split()
        
        # Very basic pattern matching for common active structures
        if len(words) >= 3:
            # Check for simple subject-verb-object pattern
            if (words[0][0].isupper() and  # Subject likely capitalized
                words[1].lower() in ['is', 'are', 'was', 'were']):
                return sentence  # Already passive-like
            
            # Simple transformation for common patterns
            if len(words) == 3:
                # "Subject Verb Object" -> "Object is Verb by Subject"
                return f"{words[2]} is {words[1]} by {words[0].lower()}"
            elif len(words) == 4:
                # Handle simple cases with articles
                if words[1] in ['a', 'an', 'the']:
                    return f"{words[2]} {words[3]} is {words[1]} by {words[0].lower()}"
        
        return sentence

    def replace_with_synonyms_simple(self, sentence):
        """
        Simplified synonym replacement without heavy models.
        Focuses on common academic words.
        """
        if not sentence:
            return sentence
            
        words = sentence.split()
        new_words = []
        
        # Common words that benefit from academic synonyms
        academic_word_map = {
            'big': ['substantial', 'considerable', 'significant'],
            'small': ['minimal', 'modest', 'limited'],
            'good': ['effective', 'beneficial', 'advantageous'],
            'bad': ['ineffective', 'detrimental', 'problematic'],
            'important': ['crucial', 'essential', 'paramount'],
            'show': ['demonstrate', 'illustrate', 'reveal'],
            'get': ['obtain', 'acquire', 'secure'],
            'use': ['utilize', 'employ', 'implement'],
            'make': ['create', 'produce', 'generate'],
            'help': ['assist', 'facilitate', 'enable'],
            'start': ['initiate', 'commence', 'undertake'],
            'end': ['conclude', 'terminate', 'complete'],
            'change': ['modify', 'alter', 'transform'],
            'look': ['examine', 'analyze', 'investigate'],
            'think': ['consider', 'contemplate', 'deliberate'],
            'know': ['understand', 'comprehend', 'recognize'],
            'see': ['observe', 'perceive', 'witness'],
            'give': ['provide', 'offer', 'supply'],
            'take': ['accept', 'receive', 'acquire'],
            'put': ['place', 'position', 'locate'],
            'keep': ['maintain', 'preserve', 'retain'],
            'let': ['allow', 'permit', 'enable'],
            'feel': ['experience', 'perceive', 'sense'],
            'try': ['attempt', 'endeavor', 'strive'],
            'work': ['function', 'operate', 'perform'],
            'need': ['require', 'necessitate', 'demand'],
            'want': ['desire', 'require', 'seek']
        }
        
        for word in words:
            # Only process words that are likely to have good synonyms
            clean_word = word.lower().strip('.,!?;:')
            
            if (len(clean_word) > 3 and 
                clean_word.isalpha() and 
                random.random() < 0.4 and  # 40% chance per eligible word
                clean_word in academic_word_map):
                
                synonyms = academic_word_map[clean_word]
                chosen_synonym = random.choice(synonyms)
                
                # Preserve capitalization
                if word[0].isupper():
                    chosen_synonym = chosen_synonym.capitalize()
                    
                new_words.append(chosen_synonym)
            else:
                new_words.append(word)
                
        return ' '.join(new_words)

    def _get_simple_synonyms(self, word):
        """
        Get synonyms using WordNet with memory limits.
        Fallback method if the static map doesn't have the word.
        """
        if not word or len(word) <= 3:
            return None
            
        try:
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    lemma_name = lemma.name().replace('_', ' ')
                    if (lemma_name.lower() != word.lower() and 
                        len(lemma_name.split()) == 1 and  # Single word only
                        lemma_name.isalpha() and
                        len(lemma_name) > 3):  # Avoid very short synonyms
                        synonyms.add(lemma_name)
                        
                        # Limit to 3 synonyms to save memory
                        if len(synonyms) >= 3:
                            break
                if len(synonyms) >= 3:
                    break
                    
            return list(synonyms) if synonyms else None
            
        except Exception:
            return None

    # Original methods kept as fallbacks but with memory optimizations
    def convert_to_passive(self, sentence):
        """
        Original passive conversion (fallback) with memory safety.
        """
        if self.nlp is None or len(sentence) > 500:
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
                        return original_str.replace(chunk, passive_str)
            return sentence
        except Exception:
            return self.convert_to_passive_simple(sentence)

    def replace_with_synonyms(self, sentence):
        """
        Original synonym replacement (fallback) with WordNet.
        """
        if len(sentence) > 1000:  # Skip for very long sentences
            return sentence
            
        try:
            tokens = word_tokenize(sentence)
            pos_tags = nltk.pos_tag(tokens)

            new_tokens = []
            for (word, pos) in pos_tags:
                if (pos.startswith(('J', 'N', 'V', 'R')) and 
                    len(word) > 3 and 
                    word.isalpha() and
                    wordnet.synsets(word)):
                    if random.random() < 0.3:  # Lower probability for memory safety
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
        """
        Get synonyms with POS filtering and memory limits.
        """
        if len(word) <= 3:
            return None
            
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
                    if (lemma_name.lower() != word.lower() and
                        len(lemma_name.split()) == 1 and
                        lemma_name.isalpha()):
                        synonyms.add(lemma_name)
                        # Strict memory limit
                        if len(synonyms) >= 5:
                            break
                if len(synonyms) >= 5:
                    break
            return list(synonyms)
        except Exception:
            return None

# Initialize NLTK resources when module is imported
download_nltk_resources()
