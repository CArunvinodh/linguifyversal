"""
AcademicTextHumanizer (ethical, backward-compatible)
- Improves fluency, variety, and naturalness of academic text.
- DOES NOT contain functionality to bypass detection systems.
- Backward-compatible init parameters:
    - p_passive (or p_passive_probability)
    - p_synonym_replacement (legacy) or p_synonym (new)
    - p_academic_transition (or p_transition)
"""

import ssl
import random
import warnings
import re
import os

import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet

import spacy

# Silence noisy warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure required NLTK resources (best-effort, quiet)
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
        except Exception:
            pass

download_nltk_resources()

# Load spaCy once (best-effort)
try:
    SPACY_NLP = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    if "sentencizer" not in SPACY_NLP.pipe_names:
        SPACY_NLP.add_pipe("sentencizer")
    # print("✅ spaCy loaded")
except Exception:
    SPACY_NLP = None
    # print("⚠️ spaCy not available")

# Helper to map NLTK tag -> WordNet
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
    Improved, ethical text humanizer.
    Backwards-compatible constructor parameter names supported:
      - p_passive or p_passive_probability
      - p_synonym_replacement (legacy) or p_synonym (preferred)
      - p_academic_transition or p_transition
    """

    def __init__(
        self,
        p_passive=0.4,
        p_synonym_replacement=None,   # legacy param name (kept for backward compat)
        p_synonym=None,               # new param name
        p_academic_transition=0.5,
        seed=None,
        use_spacy_if_available=True
    ):
        # Backwards compatibility: prefer explicit p_synonym if provided,
        # else fall back to legacy p_synonym_replacement, else default 0.6
        if p_synonym is None and p_synonym_replacement is not None:
            p_syn = p_synonym_replacement
        else:
            p_syn = p_synonym if p_synonym is not None else 0.6

        if seed is not None:
            random.seed(seed)

        # Use the global loaded spaCy model if requested and available
        self.nlp = SPACY_NLP if use_spacy_if_available else None

        # Probability params (clamped to reasonable maxima)
        self.p_passive = max(0.0, min(float(p_passive), 0.8))
        self.p_synonym = max(0.0, min(float(p_syn), 0.9))
        self.p_academic_transition = max(0.0, min(float(p_academic_transition), 0.8))

        # Curated academic transitions (used sparingly)
        self.academic_transitions = [
            "Moreover", "Additionally", "Furthermore", "However",
            "Therefore", "Consequently", "Notably", "Importantly",
            "Specifically", "In contrast", "Conversely", "Hence"
        ]

        # Contraction expansions (lowercase keys handled via regex)
        self.contractions_map = {
            "can't": "cannot", "won't": "will not", "n't": " not",
            "'re": " are", "'s": " is", "'ll": " will", "'ve": " have",
            "'d": " would", "'m": " am", "it's": "it is", "that's": "that is",
            "don't": "do not", "doesn't": "does not", "isn't": "is not"
        }

        # Curated academic vocabulary
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
        Improve text fluency and academic tone.
        - allow_passive: permit occasional safe passive transformations
        - allow_synonyms: permit context-aware synonym replacement
        Returns processed text (string).
        """

        if not isinstance(text, str):
            return "Error: input must be a string"
        text = text.strip()
        if not text:
            return ""

        if len(text) > 20000:
            return "Error: input too long (max 20,000 chars)"

        sentences = self._split_sentences(text)
        if not sentences:
            return text

        transformed = []
        for idx, sent in enumerate(sentences):
            s = sent.strip()
            if not s:
                continue

            s = self._expand_contractions(s)

            # Add academic transition sparingly (only if sentence doesn't already start with one)
            if (random.random() < self.p_academic_transition and idx % 3 == 0 and
                    not self._starts_with_transition(s)):
                s = self._add_transition(s)

            # Passive conversion (safe)
            if allow_passive and (random.random() < self.p_passive):
                s = self._safe_passive_transform(s)

            # Synonym replacement (context-aware)
            if allow_synonyms and (random.random() < self.p_synonym):
                s = self._replace_with_synonyms_context(s)

            s = self._final_clean_sentence(s)
            transformed.append(s)

        result = " ".join(transformed)
        result = self._post_process_text(result)
        return result

    # -----------------------
    # Internal helpers
    # -----------------------
    def _split_sentences(self, text):
        # Prefer spaCy segmentation, fallback to NLTK, then regex split
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

        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    def _expand_contractions(self, sentence):
        s = sentence
        # Replace longer keys first to avoid partial matches
        for k in sorted(self.contractions_map.keys(), key=len, reverse=True):
            v = self.contractions_map[k]
            pattern = re.compile(r"\b" + re.escape(k) + r"\b", flags=re.IGNORECASE)
            s = pattern.sub(v, s)
        return s

    def _starts_with_transition(self, sentence):
        words = sentence.split()
        if not words:
            return False
        head = words[0].rstrip(",").lower()
        return head in (t.lower() for t in self.academic_transitions)

    def _add_transition(self, sentence):
        t = random.choice(self.academic_transitions)
        # Capitalize sentence's first char if necessary
        if sentence and sentence[0].islower():
            sentence = sentence[0].upper() + sentence[1:]
        return f"{t}, {sentence}"

    def _safe_passive_transform(self, sentence):
        # Keep short sentences and unusual structures unchanged
        if not sentence or len(sentence.split()) < 4:
            return sentence

        # If no spaCy, avoid grammar rearrangement; append mild phrasing instead
        if not self.nlp:
            if random.random() < 0.45:
                return sentence + " This observation is supported by evidence."
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

            if root and subj and dobj and root.tag_.startswith("V"):
                verb_lemma = root.lemma_
                pp = self._past_participle_from_verb(verb_lemma)
                obj_text = self._reconstruct_span(dobj)
                subj_text = self._reconstruct_span(subj)
                if obj_text:
                    obj_text = obj_text[0].upper() + obj_text[1:]
                return f"{obj_text} is {pp} by {subj_text}."
        except Exception:
            pass

        if random.random() < 0.35:
            return sentence + " This can be observed empirically."
        return sentence

    def _past_participle_from_verb(self, lemma):
        if not lemma:
            return lemma + "ed"
        try:
            syns = wordnet.synsets(lemma, pos=wordnet.VERB)
            for syn in syns:
                for l in syn.lemmas():
                    name = l.name().replace("_", " ")
                    if name.endswith("ed") and len(name) >= len(lemma):
                        return name
        except Exception:
            pass

        if lemma.endswith("e"):
            return lemma + "d"
        if lemma.endswith("y") and len(lemma) > 2 and lemma[-2] not in "aeiou":
            return lemma[:-1] + "ied"
        return lemma + "ed"

    def _reconstruct_span(self, token):
        parts = []
        lefts = [t for t in token.lefts if t.dep_ in ("compound", "amod", "det", "nummod", "poss")]
        parts.extend([t.text for t in lefts])
        parts.append(token.text)
        rights = [t for t in token.rights if t.dep_ in ("amod", "prep", "acl")]
        for r in rights:
            if len(list(r.subtree)) <= 3:
                parts.append(r.text)
        return " ".join(parts)

    def _replace_with_synonyms_context(self, sentence):
        # Prefer spaCy path
        if self.nlp:
            try:
                doc = self.nlp(sentence)
                out_pieces = []
                for token in doc:
                    text = token.text
                    if token.is_punct or token.is_space:
                        out_pieces.append(text)
                        continue

                    if token.pos_ in ("NOUN", "VERB", "ADJ", "ADV"):
                        key = token.lemma_.lower()
                        if key in self.academic_word_map and random.random() < 0.7:
                            choice = random.choice(self.academic_word_map[key])
                            choice = self._match_case(choice, token.text)
                            out_pieces.append(choice)
                            continue

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
                                out_pieces.append(choice)
                                continue

                    out_pieces.append(text)

                # Reconstruct using token.whitespace_ to preserve spacing
                rebuilt = []
                idx = 0
                for token in doc:
                    rebuilt.append(out_pieces[idx])
                    rebuilt.append(token.whitespace_)
                    idx += 1
                return "".join(rebuilt).strip()
            except Exception:
                pass

        # Fallback (nltk-based)
        try:
            words = re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)
            alpha_indices = [i for i, w in enumerate(words) if re.match(r"\w+", w)]
            plain_words = [words[i] for i in alpha_indices]
            pos_tags = nltk.pos_tag(plain_words)
            tag_idx = 0
            new_words = []
            for i, w in enumerate(words):
                if re.match(r"\w+", w):
                    pos = pos_tags[tag_idx][1]
                    key = w.lower()
                    replaced = False
                    if key in self.academic_word_map and random.random() < 0.6:
                        choice = random.choice(self.academic_word_map[key])
                        choice = self._match_case(choice, w)
                        new_words.append(choice)
                        replaced = True
                    else:
                        wn_pos = _nltk_to_wordnet_pos(pos)
                        if wn_pos:
                            syns = self._get_simple_synonyms(key, wn_pos)
                            if syns and random.random() < 0.45:
                                choice = random.choice(syns)
                                choice = self._match_case(choice, w)
                                new_words.append(choice)
                                replaced = True
                    if not replaced:
                        new_words.append(w)
                    tag_idx += 1
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

    def _join_words_respecting_punct(self, words):
        s = " ".join(words)
        s = re.sub(r"\s+([,.;:?!])", r"\1", s)
        return s

    def _final_clean_sentence(self, s):
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"\s+,", ",", s)
        s = re.sub(r"\s+\.", ".", s)
        if s:
            s = s[0].upper() + s[1:]
        # Ensure sentence ends with punctuation
        if s and s[-1] not in ".!?":
            s = s + "."
        return s

    def _post_process_text(self, text):
        text = re.sub(r"\s+,", ",", text)
        text = re.sub(r",\s*,+", ",", text)
        text = re.sub(r"\s+\.", ".", text)
        text = re.sub(r"\s+", " ", text)
        # Prevent repeated transitions like "However, However,"
        transitions_pattern = r'(\b(?:' + "|".join([re.escape(t) for t in self.academic_transitions]) + r'),\s*){2,}'
        text = re.sub(transitions_pattern, r'\1', text, flags=re.IGNORECASE)
        return text.strip()

# ------------------------
# Example usage:
# ------------------------
if __name__ == "__main__":
    sample = (
        "The researchers use a new method. It shows that the approach works well. "
        "We don't know all the reasons, but it's important to note the results"
    )

    # Using legacy param name (p_synonym_replacement)
    hum_legacy = AcademicTextHumanizer(p_passive=0.3, p_synonym_replacement=0.6, seed=42)
    print("Legacy-param humanized:")
    print(hum_legacy.humanize_text(sample, allow_passive=True, allow_synonyms=True))

    # Using new param name (p_synonym)
    hum_new = AcademicTextHumanizer(p_passive=0.25, p_synonym=0.5, p_academic_transition=0.33, seed=7)
    print("\nNew-param humanized:")
    print(hum_new.humanize_text(sample, allow_passive=True, allow_synonyms=True))
