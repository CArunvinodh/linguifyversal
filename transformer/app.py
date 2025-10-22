import ssl
import random
import warnings
import re

import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings("ignore", category=FutureWarning)

NLP_GLOBAL = None  # Will be loaded once

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

    resources = ['punkt', 'averaged_perceptron_tagger', 'punkt_tab', 'wordnet', 'averaged_perceptron_tagger_eng']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")


class AcademicTextHumanizer:
    """
    Transforms text into a more formal (academic) style:
      - Expands contractions
      - Adds academic transitions
      - Optionally converts some sentences to passive voice
      - Optionally replaces words with synonyms for more formality
    """

    def __init__(
        self,
        model_name='paraphrase-MiniLM-L6-v2',
        p_passive=0.2,
        p_synonym_replacement=0.3,
        p_academic_transition=0.3,
        seed=None
    ):
        if seed is not None:
            random.seed(seed)

        # Load spacy model once
        global NLP_GLOBAL
        if NLP_GLOBAL is None:
            try:
                NLP_GLOBAL = spacy.load("en_core_web_sm")
            except Exception as e:
                print(f"Failed to load spacy model: {e}")
                NLP_GLOBAL = None
        
        self.nlp = NLP_GLOBAL
        
        # Model loading with fallbacks
        model_options = [
            'all-MiniLM-L6-v2',
            'sentence-transformers/all-MiniLM-L6-v2', 
            'paraphrase-MiniLM-L6-v2',
            'sentence-transformers/paraphrase-MiniLM-L6-v2'
        ]
        
        self.model = None
        for model_name_option in model_options:
            try:
                self.model = SentenceTransformer(model_name_option)
                print(f"✅ Successfully loaded model: {model_name_option}")
                break
            except Exception as e:
                print(f"❌ Failed to load {model_name_option}: {str(e)}")
                continue
        
        if self.model is None:
            print("⚠️ Using fallback mode without sentence transformers")

        # Transformation probabilities
        self.p_passive = p_passive
        self.p_synonym_replacement = p_synonym_replacement
        self.p_academic_transition = p_academic_transition

        # Common academic transitions
        self.academic_transitions = [
            "Moreover,", "Additionally,", "Furthermore,", "Hence,", 
            "Therefore,", "Consequently,", "Nonetheless,", "Nevertheless,"
        ]

    def humanize_text(self, text, use_passive=False, use_synonyms=False):
        """
        Main method to humanize text with safety limits
        """
        if not text or not text.strip():
            return ""
        
        # Safety check - prevent processing too much text
        if len(text) > 10000:
            text = text[:10000]
        
        try:
            # Use spacy for sentence splitting if available, otherwise use NLTK
            if self.nlp:
                doc = self.nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents]
            else:
                sentences = sent_tokenize(text)
            
            # Limit number of sentences to prevent memory issues
            if len(sentences) > 100:
                sentences = sentences[:100]
            
            transformed_sentences = []
            
            for sent in sentences:
                if not sent or len(sent.strip()) == 0:
                    continue
                
                sentence_str = sent.strip()
                
                # Limit individual sentence length
                if len(sentence_str) > 500:
                    sentence_str = sentence_str[:500]

                try:
                    # 1. Expand contractions
                    sentence_str = self.expand_contractions(sentence_str)

                    # 2. Possibly add academic transitions (not on every sentence)
                    if random.random() < self.p_academic_transition:
                        sentence_str = self.add_academic_transitions(sentence_str)

                    # 3. Optionally convert to passive
                    if use_passive and random.random() < self.p_passive:
                        sentence_str = self.convert_to_passive(sentence_str)

                    # 4. Optionally replace words with synonyms
                    if use_synonyms and random.random() < self.p_synonym_replacement:
                        sentence_str = self.replace_with_synonyms(sentence_str)

                    transformed_sentences.append(sentence_str)
                    
                except Exception as e:
                    # If any transformation fails, use original sentence
                    print(f"Error transforming sentence: {e}")
                    transformed_sentences.append(sent.strip())

            # Join with proper spacing
            result = ' '.join(transformed_sentences)
            
            # Final safety check on output size
            if len(result) > 20000:
                result = result[:20000]
            
            return result
            
        except Exception as e:
            print(f"Error in humanize_text: {e}")
            return text[:10000]  # Return truncated original on error

    def expand_contractions(self, sentence):
        """Expand common English contractions"""
        contraction_map = {
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'ll": " will",
            "'ve": " have",
            "'d": " would",
            "'m": " am"
        }
        
        try:
            tokens = word_tokenize(sentence)
            expanded_tokens = []
            
            for token in tokens:
                lower_token = token.lower()
                replaced = False
                
                for contraction, expansion in contraction_map.items():
                    if contraction in lower_token and lower_token.endswith(contraction):
                        new_token = lower_token.replace(contraction, expansion)
                        if token and token[0].isupper():
                            new_token = new_token.capitalize()
                        expanded_tokens.append(new_token)
                        replaced = True
                        break
                
                if not replaced:
                    expanded_tokens.append(token)

            return ' '.join(expanded_tokens)
        except Exception as e:
            print(f"Error expanding contractions: {e}")
            return sentence

    def add_academic_transitions(self, sentence):
        """Add academic transition words to sentences"""
        try:
            transition = random.choice(self.academic_transitions)
            return f"{transition} {sentence}"
        except Exception as e:
            print(f"Error adding transitions: {e}")
            return sentence

    def convert_to_passive(self, sentence):
        """Convert active voice to passive voice"""
        if not self.nlp:
            return sentence
        
        try:
            doc = self.nlp(sentence)
            subj_tokens = [t for t in doc if t.dep_ == 'nsubj' and t.head.dep_ == 'ROOT']
            dobj_tokens = [t for t in doc if t.dep_ == 'dobj']

            if subj_tokens and dobj_tokens:
                subject = subj_tokens[0]
                dobj = dobj_tokens[0]
                verb = subject.head
                
                if subject.i < verb.i < dobj.i:
                    # Create passive construction
                    passive_str = f"{dobj.text} was {verb.lemma_}ed by {subject.text}"
                    original_str = ' '.join(token.text for token in doc)
                    chunk = f"{subject.text} {verb.text} {dobj.text}"
                    
                    if chunk in original_str:
                        sentence = original_str.replace(chunk, passive_str, 1)  # Only replace first occurrence
            
            return sentence
        except Exception as e:
            print(f"Error converting to passive: {e}")
            return sentence

    def replace_with_synonyms(self, sentence):
        """Replace words with synonyms for formality"""
        try:
            tokens = word_tokenize(sentence)
            
            # Limit tokens to prevent excessive processing
            if len(tokens) > 100:
                tokens = tokens[:100]
            
            pos_tags = nltk.pos_tag(tokens)
            new_tokens = []
            
            for (word, pos) in pos_tags:
                # Only process certain POS tags and limit synonym lookups
                if pos.startswith(('J', 'N', 'V', 'R')) and len(word) > 3:
                    try:
                        synsets = wordnet.synsets(word)
                        if synsets and random.random() < 0.5:
                            synonyms = self._get_synonyms(word, pos)
                            if synonyms:
                                best_synonym = self._select_closest_synonym(word, synonyms)
                                new_tokens.append(best_synonym if best_synonym else word)
                            else:
                                new_tokens.append(word)
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
        """Get synonyms for a word based on POS tag"""
        try:
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
            synsets = wordnet.synsets(word, pos=wn_pos)
            
            # Limit number of synsets to check
            for syn in synsets[:5]:  # Only check first 5 synsets
                for lemma in syn.lemmas()[:3]:  # Only check first 3 lemmas
                    lemma_name = lemma.name().replace('_', ' ')
                    if lemma_name.lower() != word.lower():
                        synonyms.add(lemma_name)
                    
                    # Limit total synonyms
                    if len(synonyms) >= 10:
                        break
                if len(synonyms) >= 10:
                    break
            
            return list(synonyms)[:10]  # Return max 10 synonyms
        except Exception as e:
            print(f"Error getting synonyms: {e}")
            return []

    def _select_closest_synonym(self, original_word, synonyms):
        """Select the most semantically similar synonym"""
        if not synonyms:
            return None
        
        # If model failed to load, use random selection
        if self.model is None:
            return random.choice(synonyms) if synonyms else None
        
        try:
            # Limit synonyms to check
            synonyms_to_check = synonyms[:5]
            
            original_emb = self.model.encode(original_word, convert_to_tensor=True)
            synonym_embs = self.model.encode(synonyms_to_check, convert_to_tensor=True)
            cos_scores = util.cos_sim(original_emb, synonym_embs)[0]
            max_score_index = cos_scores.argmax().item()
            max_score = cos_scores[max_score_index].item()
            
            # Only use synonym if similarity is high enough
            if max_score >= 0.5:
                return synonyms_to_check[max_score_index]
            return None
        except Exception as e:
            # Fallback to random selection if encoding fails
            print(f"Model encoding failed, using random synonym: {e}")
            return random.choice(synonyms) if synonyms else None
