"""
AcademicTextHumanizer (ethical)
- Purpose: improve fluency, variety, and naturalness of academic text.
- NOT for evading plagiarism / AI-detection systems or cheating.
- Use to edit and improve your own writing; always cite sources and follow academic integrity.
"""

import ssl
import random
import warnings
import re
import os

# NLP libraries
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet

import spacy

# Disable noisy warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure required NLTK resources are present (quiet)
def download_nltk_resources():
    try:
        _create_unverified = ssl._create_unverified_context
    except Exception:
        pass
    else:
        ssl._create_default_https_context = _create_unverified

    resources = ["punkt", "averaged_perceptron_tagger", "wordnet"]
    for r in resources:
        try:
            nltk.download(r, quiet=True)
        except Exception as e:
            # Best-effort download; continue if offline
            print(f"⚠️ NLTK resource {r} download failed or not necessary: {e}")

download_nltk_resources()

# Try loading spaCy once at module load (best-effort)
try:
    SPACY_NLP = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    if "sentencizer" not in SPACY_NLP.pipe_names:
        SPACY_NLP.add_pipe("sentencizer")
    print("✅ spaCy loaded (en_core_web_sm)")
except Exception:
    SPACY_NLP = None
    print("⚠️ spaCy 'en_core_web_sm' not available; will use lighter fallbacks")

# Helper: map nltk pos tag to wordnet pos
def _nltk_to_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return None

class AcademicTextHumanizer:
    """
    Improved humanizer for academic text. Ethical use only.
    """

    def __init__(
        self,
        p_passive=0.25,
        p_synonym=0.45,
        p_transition=0.33,
        seed=None,
        use_spacy_if_available=True
    ):
        if seed is not None:
            random.seed(seed)

        # Load spaCy per-instance if available and desired
        self.nlp = SPACY_NLP if use_spacy_if_available else None

        # Probabilities (kept moderate)
        self.p_passive = float(p_passive)
        self.p_synonym = float(p_synonym)
        self.p_transition = float(p_transition)

        # Academic transitions - used sparingly and contextually
        self.academic_transitions = [
            "Moreover", "Additionally", "Furthermore", "However",
            "Therefore", "Consequently", "Notably", "Importantly",
            "Specifically", "In contrast", "Conversely", "Hence"
        ]

        # Contractions map - lowercased keys; expansion uses regex word boundaries
        self.contractions_map = {
            "can't": "cannot", "won't": "will not", "n't": " not",
            "'re": " are", "'s": " is", "'ll": " will", "'ve": " have",
            "'d": " would", "'m": " am", "it's": "it is", "that's": "that is",
            "don't": "do not", "doesn't": "does not", "isn't": "is not"
        }

        # Static academic vocabulary (smaller, curated)
        self.academic_word_map = {
            "use": ["utilize", "employ", "apply", "leverage"],
            "make": ["produce", "create", "construct"],
            "get": ["obtain", "acquire", "attain"],
            "show": ["demonstrate", "illustrate", "reveal"],
            "help": ["assist", "facilitate", "support"],
            "start": ["initiate", "commence", "undertake"],
            "change": ["modify", "alter", "transform"],
            "good": ["effective", "beneficial", "advantageous"],
            "important": ["crucial", "essential", "paramount"],
            "problem": ["issue", "challenge", "obstacle"],
            "way": ["method", "approach", "technique"],
            "result": ["outcome", "finding", "consequence"]
        }

    # -----------------------
    # Public API
    # -----------------------
    def humanize_text(self, text, *, allow_passive=False, allow_synonyms=True):
        """
        Main entry point.
        - allow_passive: permit occasional (safe) passive-style rewrites
        - allow_synonyms: allow context-aware synonym replacement
        Returns improved text or error string.
        """
        # Basic validation
        if not isinstance(text, str):
            return "Error: input must be a string."
        text = text.strip()
        if not text:
            return ""

        if len(text) > 20000:
            return "Error: input too long (max 20,000 characters)."

        # Sentence splitting (prefer spaCy segmentation if available)
        sentences = self._split_sentences(text)
        if not sentences:
            return text

        transformed = []
        for idx, sent in enumerate(sentences):
            s = sent.strip()
            if not s:
                continue

            s = self._expand_contractions(s)

            # Optionally add an academic transition, only if sentence doesn't already start with one
            if (random.random() < self.p_transition and idx % 3 == 0 and
                    not self._starts_with_transition(s)):
                s = self._add_transition(s)

            # Passive conversion: attempt safe pattern-aware transform
            if allow_passive and (random.random() < self.p_passive):
                s = self._safe_passive_transform(s)

            # Synonym replacement: POS-aware
            if allow_synonyms and (random.random() < self.p_synonym):
                s = self._replace_with_synonyms_context(s)

            # Clean spacing and punctuation
            s = self._final_clean_sentence(s)
            transformed.append(s)

        result = " ".join(transformed)
        result = self._post_process_text(result)
        return result

    # -----------------------
    # Internal helpers
    # -----------------------
    def _split_sentences(self, text):
        # Prefer spaCy, fallback to NLTK's sent_tokenize, last fallback: simple split.
        if self.nlp:
            try:
                doc = self.nlp(text)
                sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
                if sentences:
                    return sentences
            except Exception:
                pass

        try:
            sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
            if sentences:
                return sentences
        except Exception:
            pass

        # Fallback: split on punctuation
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    def _expand_contractions(self, sentence):
        # Use regex word boundaries to avoid accidental replacements
        s = sentence
        # Sort by length to replace longer keys first
        for k in sorted(self.contractions_map.keys(), key=len, reverse=True):
            v = self.contractions_map[k]
            pattern = re.compile(r"\b" + re.escape(k) + r"\b", flags=re.IGNORECASE)
            s = pattern.sub(v, s)
        return s

    def _starts_with_transition(self, sentence):
        # Check if sentence already starts with a known transition (case-insensitive)
        head = sentence.split()[0].rstrip(",")
        return head.lower() in (t.lower() for t in self.academic_transitions)

    def _add_transition(self, sentence):
        # Choose a transition that fits punctuation conventions
        t = random.choice(self.academic_transitions)
        # If sentence starts with lowercase, capitalize first letter after transition
        if sentence and sentence[0].islower():
            sentence = sentence[0].upper() + sentence[1:]
        return f"{t}, {sentence}"

    def _safe_passive_transform(self, sentence):
        """
        Attempt a simple, safe passive transformation using spaCy dependency parse
        if available. If parse isn't reliable or no clear subject/object, return
        original sentence unchanged.
        Note: This is meant for stylistic variation only, not to obfuscate origin.
        """
        if not sentence or len(sentence.split()) < 4:
            return sentence

        if not self.nlp:
            # Heuristic fallback: append an academic phrasing instead of altering grammar
            if random.random() < 0.5:
                return sentence + " This observation is supported by the evidence."
            return sentence

        try:
            doc = self.nlp(sentence)
            root = None
            subj = None
            dobj = None
            for token in doc:
                if token.dep_ == "ROOT":
                    root = token
                if token.dep_ in ("nsubj", "nsubjpass") and subj is None:
                    subj = token
                if token.dep_ in ("dobj", "obj") and dobj is None:
                    dobj = token

            # Proceed only if we have a subject, a root verb, and an object
            if root and subj and dobj and root.tag_.startswith("V"):
                # Get base form and past participle where possible using WordNet or simple heuristics
                verb_lemma = root.lemma_
                pp = self._past_participle_from_verb(verb_lemma)
                obj_text = self._reconstruct_span(dobj)
                subj_text = self._reconstruct_span(subj)
                # Keep capitalization consistent
                obj_text_cap = obj_text[0].upper() + obj_text[1:] if obj_text else obj_text
                return f"{obj_text_cap} is {pp} by {subj_text}."
        except Exception:
            pass

        # Conservative fallback identity or mild academic addition
        if random.random() < 0.35:
            return sentence + " This can be observed empirically."
        return sentence

    def _past_participle_from_verb(self, lemma):
        """
        Try to find a reasonable past participle for a verb lemma:
        - Check WordNet for verb forms (limited).
        - Apply common regular rules with safeguards.
        """
        if not lemma:
            return lemma + "ed"

        # quick WordNet search for lemmatized synset forms
        try:
            syns = wordnet.synsets(lemma, pos=wordnet.VERB)
            for syn in syns:
                for l in syn.lemmas():
                    name = l.name().replace("_", " ")
                    # naive attempt: if lemma differs and ends with 'ed' or common participle
                    if len(name) > len(lemma) and name.endswith("ed"):
                        return name
        except Exception:
            pass

        # Regular heuristics
        if lemma.endswith("e"):
            return lemma + "d"
        if lemma.endswith("y") and len(lemma) > 2 and lemma[-2] not in "aeiou":
            return lemma[:-1] + "ied"
        # Default: add 'ed' (may be imperfect but safe)
        return lemma + "ed"

    def _reconstruct_span(self, token):
        """Return a simple span text for token (including compound modifiers)."""
        # Try to include adjectival modifiers and compounds to form reasonable phrase
        parts = []
        # left children that are compounds or modifiers
        lefts = [t for t in token.lefts if t.dep_ in ("compound", "amod", "det", "nummod", "poss")]
        parts.extend([t.text for t in lefts])
        parts.append(token.text)
        rights = [t for t in token.rights if t.dep_ in ("amod", "prep", "acl")]
        # avoid long prepositional attachments; only include short ones if safe
        for r in rights:
            if len(list(r.subtree)) <= 3:
                parts.append(r.text)
        return " ".join(parts)

    def _replace_with_synonyms_context(self, sentence):
        """
        POS-aware synonym replacement:
        - If spaCy available: use token.pos_ to decide what to replace.
        - Otherwise: use naive word-level replacement with NLTK POS tags.
        - Use static academic map first, then WordNet fallback with POS matching.
        """
        # If spaCy available, use it for careful tokenization & POS
        if self.nlp:
            try:
                doc = self.nlp(sentence)
                out_tokens = []
                for token in doc:
                    text = token.text
                    # keep punctuation untouched
                    if token.is_punct or token.is_space:
                        out_tokens.append(text)
                        continue

                    # Candidate words: NOUN, VERB, ADJ, ADV
                    if token.pos_ in ("NOUN", "VERB", "ADJ", "ADV"):
                        key = token.lemma_.lower()
                        # Prefer curated academic map on lemma
                        if key in self.academic_word_map and random.random() < 0.7:
                            choice = random.choice(self.academic_word_map[key])
                            choice = self._match_case(choice, token.text)
                            out_tokens.append(choice)
                            continue

                        # WordNet fallback: attempt to fetch synonyms with matching POS
                        wn_pos = None
                        if token.pos_ == "NOUN":
                            wn_pos = wordnet.NOUN
                        elif token.pos_ == "VERB":
                            wn_pos = wordnet.VERB
                        elif token.pos_ == "ADJ":
                            wn_pos = wordnet.ADJ
                        elif token.pos_ == "ADV":
                            wn_pos = wordnet.ADV

                        if wn_pos:
                            syns = self._get_simple_synonyms(token.lemma_, wn_pos)
                            if syns and random.random() < 0.5:
                                choice = random.choice(syns)
                                choice = self._match_case(choice, token.text)
                                out_tokens.append(choice)
                                continue

                    # Default: keep original token text
                    out_tokens.append(text)
                # Join preserving whitespace/punctuation: spaCy tokens include spacing attribute
                result = "".join([t if getattr(t, "isspace", False) else t for t in out_tokens])
                # If join created awkward spacing (common), reconstruct from tokens:
                return self._reconstruct_from_spacy_tokens(doc, out_tokens)
            except Exception:
                pass

        # Fallback: naive approach using regex word boundaries and NLTK POS
        try:
            words = re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)
            pos_tags = nltk.pos_tag([w for w in words if re.match(r"\w+", w)])
            tag_map = {}
            idx = 0
            for i, w in enumerate(words):
                if re.match(r"\w+", w):
                    tag_map[i] = pos_tags[idx][1]
                    idx += 1

            new_words = []
            for i, w in enumerate(words):
                if re.match(r"\w+", w):
                    pos = tag_map.get(i, "")
                    key = w.lower()
                    replaced = False
                    if key in self.academic_word_map and random.random() < 0.6:
                        choice = random.choice(self.academic_word_map[key])
                        choice = self._match_case(choice, w)
                        new_words.append(choice)
                        replaced = True
                    else:
                        wn_pos = _nltk_to_wordnet_pos(pos) or None
                        if wn_pos:
                            syns = self._get_simple_synonyms(key, wn_pos)
                            if syns and random.random() < 0.45:
                                choice = random.choice(syns)
                                choice = self._match_case(choice, w)
                                new_words.append(choice)
                                replaced = True
                    if not replaced:
                        new_words.append(w)
                else:
                    new_words.append(w)
            return self._join_words_respecting_punct(new_words)
        except Exception:
            return sentence

    def _match_case(self, candidate, original):
        if original.istitle():
            return candidate.capitalize()
        if original.isupper():
            return candidate.upper()
        return candidate

    def _get_simple_synonyms(self, lemma, wn_pos):
        """
        Query WordNet for synonyms of lemma with given POS; keep short list.
        """
        try:
            synonyms = set()
            for syn in wordnet.synsets(lemma, pos=wn_pos):
                for l in syn.lemmas():
                    name = l.name().replace("_", " ")
                    if name.lower() != lemma.lower() and name.isalpha() and len(name) > 3:
                        synonyms.add(name)
                        if len(synonyms) >= 4:
                            break
                if len(synonyms) >= 4:
                    break
            return list(synonyms) if synonyms else None
        except Exception:
            return None

    def _reconstruct_from_spacy_tokens(self, doc, out_tokens):
        """
        Reconstruct text after token-level replacement while preserving spacing.
        This uses spaCy token.whitespace_ where available.
        """
        pieces = []
        i = 0
        for token in doc:
            replacement = out_tokens[i]
            pieces.append(replacement)
            # spaCy token.whitespace_ contains the following whitespace (possibly empty)
            pieces.append(token.whitespace_)
            i += 1
        return "".join(pieces).strip()

    def _join_words_respecting_punct(self, words):
        # Join words while avoiding space before punctuation
        s = " ".join(words)
        s = re.sub(r"\s+([,.;:?!])", r"\1", s)
        return s

    def _final_clean_sentence(self, s):
        # Basic fixes: collapse multiple spaces, ensure spacing after commas, etc.
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"\s+,", ",", s)
        s = re.sub(r"\s+\.", ".", s)
        # Ensure first char capitalization of the sentence
        if s:
            s = s[0].upper() + s[1:]
        return s

    def _post_process_text(self, text):
        # Remove duplicated transitions and tidy punctuation spacing
        # Collapse multiple commas, unify spacing
        text = re.sub(r"\s+,", ",", text)
        text = re.sub(r",\s*,+", ",", text)
        text = re.sub(r"\s+\.", ".", text)
        text = re.sub(r"\s+", " ", text)
        # Avoid sequences like "However, However," by limiting repeated starters
        text = re.sub(r'(\b(?:' + "|".join([re.escape(t) for t in self.academic_transitions]) + r'),\s*){2,}', r'\1', text, flags=re.IGNORECASE)
        return text.strip()


# Example usage (for testing) - remove or wrap under __main__ in production
if __name__ == "__main__":
    sample = (
        "The researchers use a new method. It shows that the approach works well. "
        "We don't know all the reasons, but it's important to note the results."
    )
    hum = AcademicTextHumanizer(seed=42)
    print("Original:\n", sample)
    print("\nHumanized:\n", hum.humanize_text(sample, allow_passive=True, allow_synonyms=True))
