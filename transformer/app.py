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
        p_passive=0.4,  # Increased probabilities
        p_synonym_replacement=0.6,
        p_academic_transition=0.5,
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

        # Increased probabilities for better transformations
        self.p_passive = min(p_passive, 0.6)  # Increased cap
        self.p_synonym_replacement = min(p_synonym_replacement, 0.8)
        self.p_academic_transition = min(p_academic_transition, 0.6)

        # Enhanced academic transitions
        self.academic_transitions = [
            "Moreover,", "Additionally,", "Furthermore,", "Hence,", 
            "Therefore,", "Consequently,", "Nonetheless,", "Nevertheless,",
            "Accordingly,", "Thus,", "Notably,", "Significantly,",
            "Importantly,", "Specifically,", "Typically,", "Generally,"
        ]

        # Enhanced contractions mapping
        self.contractions_map = {
            "n't": " not", "'re": " are", "'s": " is", "'ll": " will",
            "'ve": " have", "'d": " would", "'m": " am",
            "can't": "cannot", "won't": "will not", "don't": "do not",
            "doesn't": "does not", "isn't": "is not", "aren't": "are not",
            "wasn't": "was not", "weren't": "were not", "haven't": "have not",
            "hasn't": "has not", "hadn't": "had not", "wouldn't": "would not",
            "shouldn't": "should not", "couldn't": "could not", "mightn't": "might not",
            "mustn't": "must not", "shan't": "shall not", "ain't": "am not",
            "it's": "it is", "that's": "that is", "what's": "what is",
            "who's": "who is", "where's": "where is", "when's": "when is",
            "why's": "why is", "how's": "how is", "there's": "there is",
            "here's": "here is", "everybody's": "everybody is", "everyone's": "everyone is",
            "nobody's": "nobody is", "someone's": "someone is", "something's": "something is"
        }

    def humanize_text(self, text, use_passive=False, use_synonyms=False):
        """
        Improved humanization with better transformations and higher probabilities
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

                # 1. Expand contractions (always applied)
                current_sentence = self.expand_contractions(current_sentence)

                # 2. Add academic transitions with higher probability
                if random.random() < self.p_academic_transition:
                    current_sentence = self.add_academic_transitions(current_sentence)

                # 3. Convert to passive if enabled
                if use_passive and random.random() < self.p_passive:
                    current_sentence = self.convert_to_passive_improved(current_sentence)

                # 4. Replace with synonyms if enabled
                if use_synonyms and random.random() < self.p_synonym_replacement:
                    current_sentence = self.replace_with_synonyms_improved(current_sentence)

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
        Comprehensive contraction expansion.
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

    def convert_to_passive_improved(self, sentence):
        """
        Improved passive voice conversion with better pattern matching.
        """
        if not sentence or len(sentence.split()) < 3:
            return sentence

        words = sentence.split()
        
        # Enhanced verb to past participle mapping
        verb_map = {
            'uses': 'used', 'use': 'used', 'using': 'used',
            'creates': 'created', 'create': 'created', 'creating': 'created',
            'develops': 'developed', 'develop': 'developed', 'developing': 'developed',
            'implements': 'implemented', 'implement': 'implemented', 'implementing': 'implemented',
            'builds': 'built', 'build': 'built', 'building': 'built',
            'makes': 'made', 'make': 'made', 'making': 'made',
            'takes': 'taken', 'take': 'taken', 'taking': 'taken',
            'gives': 'given', 'give': 'given', 'giving': 'given',
            'shows': 'shown', 'show': 'shown', 'showing': 'shown',
            'sees': 'seen', 'see': 'seen', 'seeing': 'seen',
            'finds': 'found', 'find': 'found', 'finding': 'found',
            'provides': 'provided', 'provide': 'provided', 'providing': 'provided',
            'offers': 'offered', 'offer': 'offered', 'offering': 'offered',
            'achieves': 'achieved', 'achieve': 'achieved', 'achieving': 'achieved',
            'obtains': 'obtained', 'obtain': 'obtained', 'obtaining': 'obtained',
            'produces': 'produced', 'produce': 'produced', 'producing': 'produced',
            'generates': 'generated', 'generate': 'generated', 'generating': 'generated'
        }
        
        # Handle different sentence structures
        if len(words) >= 3:
            # Pattern: Subject + Verb + Object
            subject = words[0]
            verb = words[1]
            object_words = ' '.join(words[2:])
            
            # Get past participle
            past_participle = verb_map.get(verb.lower())
            if not past_participle:
                # Try to create past participle for regular verbs
                if verb.endswith('s'):
                    past_participle = verb[:-1] + 'ed'
                elif verb.endswith('ing'):
                    past_participle = verb[:-3] + 'ed'
                else:
                    past_participle = verb + 'ed'
            
            # Apply passive transformation
            if verb.lower() in verb_map or verb.endswith('s') or verb.endswith('ing'):
                # Handle articles
                if subject.lower() in ['a', 'an', 'the'] and len(words) > 3:
                    subject = f"{subject} {words[1]}"
                    verb = words[2]
                    object_words = ' '.join(words[3:])
                    past_participle = verb_map.get(verb.lower(), verb + 'ed')
                
                return f"{object_words} is {past_participle} by {subject.lower()}"
        
        return sentence

    def replace_with_synonyms_improved(self, sentence):
        """
        Enhanced synonym replacement with comprehensive academic vocabulary.
        """
        if not sentence:
            return sentence
            
        words = sentence.split()
        new_words = []
        
        # Comprehensive academic vocabulary mapping
        academic_word_map = {
            # Common verbs
            'use': ['utilize', 'employ', 'implement', 'apply', 'leverage'],
            'make': ['create', 'produce', 'generate', 'construct', 'fabricate'],
            'get': ['obtain', 'acquire', 'secure', 'procure', 'attain'],
            'give': ['provide', 'offer', 'supply', 'furnish', 'deliver'],
            'show': ['demonstrate', 'illustrate', 'reveal', 'exhibit', 'manifest'],
            'help': ['assist', 'facilitate', 'enable', 'support', 'aid'],
            'start': ['initiate', 'commence', 'begin', 'undertake', 'launch'],
            'change': ['modify', 'alter', 'transform', 'adjust', 'adapt'],
            'keep': ['maintain', 'preserve', 'retain', 'sustain', 'uphold'],
            'put': ['place', 'position', 'locate', 'situate', 'deposit'],
            'take': ['accept', 'receive', 'acquire', 'adopt', 'appropriate'],
            'think': ['consider', 'contemplate', 'deliberate', 'reflect', 'ponder'],
            'know': ['understand', 'comprehend', 'recognize', 'apprehend', 'discern'],
            'see': ['observe', 'perceive', 'witness', 'discern', 'behold'],
            'want': ['desire', 'require', 'seek', 'request', 'solicit'],
            
            # Adjectives
            'good': ['effective', 'beneficial', 'advantageous', 'favorable', 'superior'],
            'bad': ['ineffective', 'detrimental', 'problematic', 'unfavorable', 'deleterious'],
            'big': ['substantial', 'considerable', 'significant', 'extensive', 'substantive'],
            'small': ['minimal', 'modest', 'limited', 'negligible', 'insubstantial'],
            'important': ['crucial', 'essential', 'paramount', 'critical', 'vital'],
            'different': ['distinct', 'disparate', 'varied', 'diverse', 'heterogeneous'],
            'new': ['novel', 'innovative', 'recent', 'contemporary', 'modern'],
            'old': ['traditional', 'conventional', 'established', 'time-honored', 'archaic'],
            
            # Nouns
            'way': ['method', 'approach', 'technique', 'strategy', 'methodology'],
            'thing': ['aspect', 'element', 'component', 'factor', 'entity'],
            'stuff': ['material', 'substance', 'content', 'matter', 'composition'],
            'problem': ['issue', 'challenge', 'difficulty', 'obstacle', 'complication'],
            'answer': ['solution', 'resolution', 'explanation', 'clarification', 'response'],
            'job': ['task', 'assignment', 'responsibility', 'duty', 'obligation'],
            
            # Academic phrases
            'a lot of': ['numerous', 'multiple', 'various', 'considerable', 'substantial'],
            'lots of': ['abundant', 'plentiful', 'copious', 'substantial', 'profuse'],
            'kind of': ['somewhat', 'rather', 'moderately', 'relatively', 'comparatively'],
            'sort of': ['somewhat', 'rather', 'moderately', 'relatively', 'comparatively']
        }
        
        i = 0
        while i < len(words):
            word = words[i]
            clean_word = word.lower().strip('.,!?;:"')
            
            # Check for multi-word phrases first
            if i < len(words) - 1:
                two_word = f"{clean_word} {words[i+1].lower().strip('.,!?;:"')}"
                if two_word in academic_word_map:
                    synonyms = academic_word_map[two_word]
                    chosen_synonym = random.choice(synonyms)
                    new_words.append(chosen_synonym)
                    i += 2  # Skip next word
                    continue
            
            # Single word replacement
            if (len(clean_word) > 2 and 
                clean_word.isalpha() and 
                random.random() < 0.7 and  # High probability for eligible words
                clean_word in academic_word_map):
                
                synonyms = academic_word_map[clean_word]
                chosen_synonym = random.choice(synonyms)
                
                # Preserve capitalization
                if word[0].isupper():
                    chosen_synonym = chosen_synonym.capitalize()
                    
                new_words.append(chosen_synonym)
            else:
                new_words.append(word)
            
            i += 1
    
        # Remove any empty strings and join
        result = ' '.join([w for w in new_words if w])
        return result

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

# Initialize NLTK resources when module is imported
download_nltk_resources()
