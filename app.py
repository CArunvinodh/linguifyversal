"""
Lightweight Academic Text Humanizer
No spacy, no sentence-transformers - pure NLTK only
Optimized for serverless deployment
"""

import ssl
import random
import warnings
import re

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet

warnings.filterwarnings("ignore")

def download_nltk_resources():
    """Download required NLTK resources"""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    resources = ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")


class AcademicTextHumanizer:
    """
    Lightweight text transformer using only NLTK
    - Expands contractions
    - Adds academic transitions
    - Simple passive voice conversion
    - Synonym replacement using WordNet
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

        self.p_passive = p_passive
        self.p_synonym_replacement = p_synonym_replacement
        self.p_academic_transition = p_academic_transition

        self.academic_transitions = [
            "Moreover,", "Additionally,", "Furthermore,", "Hence,",
            "Therefore,", "Consequently,", "Nonetheless,", "Nevertheless,",
            "In addition,", "Similarly,", "Conversely,", "Subsequently,"
        ]

        # Simple passive voice patterns
        self.common_verbs = {
            'make': 'made', 'take': 'taken', 'give': 'given',
            'find': 'found', 'show': 'shown', 'use': 'used',
            'see': 'seen', 'know': 'known', 'get': 'gotten',
            'write': 'written', 'read': 'read', 'hear': 'heard'
        }

    def humanize_text(self, text, use_passive=False, use_synonyms=False):
        """Main transformation method"""
        if not text or not text.strip():
            return ""
        
        # Safety limits
        if len(text) > 10000:
            text = text[:10000]
        
        try:
            sentences = sent_tokenize(text)
            
            # Limit sentences
            if len(sentences) > 100:
                sentences = sentences[:100]
            
            transformed_sentences = []
            
            for i, sent in enumerate(sentences):
                if not sent or len(sent.strip()) == 0:
                    continue
                
                sentence_str = sent.strip()
                
                # Limit sentence length
                if len(sentence_str) > 500:
                    sentence_str = sentence_str[:500]

                try:
                    # 1. Expand contractions
                    sentence_str = self.expand_contractions(sentence_str)

                    # 2. Add academic transitions (not on first sentence, not on every sentence)
                    if i > 0 and random.random() < self.p_academic_transition:
                        sentence_str = self.add_academic_transitions(sentence_str)

                    # 3. Convert to passive voice
                    if use_passive and random.random() < self.p_passive:
                        sentence_str = self.convert_to_passive_simple(sentence_str)

                    # 4. Replace with synonyms
                    if use_synonyms and random.random() < self.p_synonym_replacement:
                        sentence_str = self.replace_with_synonyms(sentence_str)

                    transformed_sentences.append(sentence_str)
                    
                except Exception as e:
                    print(f"Error transforming sentence: {e}")
                    transformed_sentences.append(sent.strip())

            result = ' '.join(transformed_sentences)
            
            # Final safety check
            if len(result) > 20000:
                result = result[:20000]
            
            return result
            
        except Exception as e:
            print(f"Error in humanize_text: {e}")
            return text[:10000]

    def expand_contractions(self, sentence):
        """Expand common English contractions"""
        contraction_map = {
            "n't": " not", "'re": " are", "'s": " is", "'ll": " will",
            "'ve": " have", "'d": " would", "'m": " am",
            "won't": "will not", "can't": "cannot", "ain't": "am not"
        }
        
        try:
            result = sentence
            for contraction, expansion in contraction_map.items():
                # Case-insensitive replacement
                pattern = re.compile(re.escape(contraction), re.IGNORECASE)
                result = pattern.sub(expansion, result)
            return result
        except Exception as e:
            print(f"Error expanding contractions: {e}")
            return sentence

    def add_academic_transitions(self, sentence):
        """Add academic transition words"""
        try:
            transition = random.choice(self.academic_transitions)
            return f"{transition} {sentence}"
        except Exception:
            return sentence

    def convert_to_passive_simple(self, sentence):
        """
        Simple passive voice conversion using pattern matching
        Only works for simple Subject-Verb-Object patterns
        """
        try:
            tokens = word_tokenize(sentence)
            pos_tags = nltk.pos_tag(tokens)
            
            # Look for simple SVO pattern: NOUN VERB NOUN
            for i in range(len(pos_tags) - 2):
                if (pos_tags[i][1].startswith('NN') and 
                    pos_tags[i+1][1].startswith('VB') and 
                    pos_tags[i+2][1].startswith('NN')):
                    
                    subject = pos_tags[i][0]
                    verb = pos_tags[i+1][0]
                    obj = pos_tags[i+2][0]
                    
                    # Get past participle form
                    verb_base = verb.lower()
                    if verb_base in self.common_verbs:
                        past_part = self.common_verbs[verb_base]
                    elif verb_base.endswith('e'):
                        past_part = verb_base + 'd'
                    else:
                        past_part = verb_base + 'ed'
                    
                    # Create passive: Object was verb+ed by subject
                    passive = f"{obj} was {past_part} by {subject}"
                    
                    # Replace in original sentence
                    original_phrase = f"{subject} {verb} {obj}"
                    sentence = sentence.replace(original_phrase, passive, 1)
                    break
            
            return sentence
        except Exception as e:
            print(f"Error in passive conversion: {e}")
            return sentence

    def replace_with_synonyms(self, sentence):
        """Replace words with formal synonyms using WordNet"""
        try:
            tokens = word_tokenize(sentence)
            
            if len(tokens) > 100:
                tokens = tokens[:100]
            
            pos_tags = nltk.pos_tag(tokens)
            new_tokens = []
            
            for word, pos in pos_tags:
                # Only process content words longer than 3 characters
                if (pos.startswith(('J', 'N', 'V', 'R')) and 
                    len(word) > 3 and 
                    word.isalpha() and
                    random.random() < 0.4):  # 40% chance to replace
                    
                    try:
                        synonyms = self._get_synonyms(word, pos)
                        if synonyms:
                            # Pick a random synonym
                            synonym = random.choice(synonyms)
                            # Preserve capitalization
                            if word[0].isupper():
                                synonym = synonym.capitalize()
                            new_tokens.append(synonym)
                        else:
                            new_tokens.append(word)
                    except Exception:
                        new_tokens.append(word)
                else:
                    new_tokens.append(word)

            return ' '.join(new_tokens)
        except Exception as e:
            print(f"Error replacing synonyms: {e}")
            return sentence

    def _get_synonyms(self, word, pos):
        """Get synonyms from WordNet"""
        try:
            # Map POS tags to WordNet POS
            wn_pos = None
            if pos.startswith('J'):
                wn_pos = wordnet.ADJ
            elif pos.startswith('N'):
                wn_pos = wordnet.NOUN
            elif pos.startswith('R'):
                wn_pos = wordnet.ADV
            elif pos.startswith('V'):
                wn_pos = wordnet.VERB
            else:
                return []

            synonyms = []
            synsets = wordnet.synsets(word.lower(), pos=wn_pos)
            
            # Check first 3 synsets only
            for syn in synsets[:3]:
                for lemma in syn.lemmas()[:3]:
                    lemma_name = lemma.name().replace('_', ' ')
                    # Only use single-word synonyms that are different
                    if ' ' not in lemma_name and lemma_name.lower() != word.lower():
                        synonyms.append(lemma_name)
                    
                    if len(synonyms) >= 5:
                        break
                if len(synonyms) >= 5:
                    break
            
            return synonyms[:5]
        except Exception as e:
            print(f"Error getting synonyms: {e}")
            return []
