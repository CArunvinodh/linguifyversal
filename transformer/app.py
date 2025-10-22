import ssl
import random
import warnings
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
import spacy

# -------------------
# NLTK & spaCy setup
# -------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def download_nltk_resources():
    try:
        _create_unverified = ssl._create_unverified_context
        ssl._create_default_https_context = _create_unverified
    except Exception:
        pass

    for r in ["punkt", "averaged_perceptron_tagger", "wordnet"]:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass

download_nltk_resources()

try:
    SPACY_NLP = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    if "sentencizer" not in SPACY_NLP.pipe_names:
        SPACY_NLP.add_pipe("sentencizer")
except Exception:
    SPACY_NLP = None

def _nltk_to_wordnet_pos(tag):
    if tag.startswith("J"): return wordnet.ADJ
    if tag.startswith("V"): return wordnet.VERB
    if tag.startswith("N"): return wordnet.NOUN
    if tag.startswith("R"): return wordnet.ADV
    return None

# -------------------
# Main Class
# -------------------
class AcademicTextHumanizer:
    def __init__(
        self,
        p_passive=0.4,
        p_synonym_replacement=None,  # legacy name
        p_synonym=None,              # new name
        p_academic_transition=0.5,
        seed=None,
        use_spacy_if_available=True
    ):
        if p_synonym is None and p_synonym_replacement is not None:
            p_syn = p_synonym_replacement
        else:
            p_syn = p_synonym if p_synonym is not None else 0.6

        if seed is not None:
            random.seed(seed)

        self.nlp = SPACY_NLP if use_spacy_if_available else None
        self.p_passive = max(0.0, min(float(p_passive), 0.8))
        self.p_synonym = max(0.0, min(float(p_syn), 0.9))
        self.p_academic_transition = max(0.0, min(float(p_academic_transition), 0.8))

        self.academic_transitions = [
            "Moreover", "Additionally", "Furthermore", "However", "Therefore",
            "Consequently", "Notably", "Importantly", "Specifically",
            "In contrast", "Conversely", "Hence"
        ]

        self.contractions_map = {
            "can't": "cannot", "won't": "will not", "n't": " not",
            "'re": " are", "'s": " is", "'ll": " will", "'ve": " have",
            "'d": " would", "'m": " am", "it's": "it is", "that's": "that is",
            "don't": "do not", "doesn't": "does not", "isn't": "is not"
        }

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

    # -------------------
    # Main method
    # -------------------
    def humanize_text(
        self,
        text,
        *,
        allow_passive=False,
        allow_synonyms=True,
        **kwargs
    ):
        """
        Backward compatible:
        - accepts allow_passive or use_passive
        - accepts allow_synonyms or use_synonyms
        """
        # Backward-compat alias support
        if "use_passive" in kwargs:
            allow_passive = kwargs["use_passive"]
        if "use_synonyms" in kwargs:
            allow_synonyms = kwargs["use_synonyms"]

        if not isinstance(text, str):
            return "Error: input must be a string"
        text = text.strip()
        if not text:
            return ""

        sentences = self._split_sentences(text)
        transformed = []
        for idx, s in enumerate(sentences):
            s = s.strip()
            if not s:
                continue

            s = self._expand_contractions(s)
            if random.random() < self.p_academic_transition and idx % 3 == 0:
                s = self._add_transition(s)

            if allow_passive and random.random() < self.p_passive:
                s = self._safe_passive_transform(s)
            if allow_synonyms and random.random() < self.p_synonym:
                s = self._replace_with_synonyms_context(s)

            transformed.append(self._final_clean_sentence(s))

        result = " ".join(transformed)
        return self._post_process_text(result)

    # -------------------
    # Helpers
    # -------------------
    def _split_sentences(self, text):
        if self.nlp:
            try:
                doc = self.nlp(text)
                return [s.text.strip() for s in doc.sents if s.text.strip()]
            except Exception:
                pass
        try:
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except Exception:
            return re.split(r'(?<=[.!?])\s+', text)

    def _expand_contractions(self, sentence):
        s = sentence
        for k in sorted(self.contractions_map.keys(), key=len, reverse=True):
            v = self.contractions_map[k]
            s = re.sub(rf"\b{re.escape(k)}\b", v, s, flags=re.IGNORECASE)
        return s

    def _add_transition(self, sentence):
        t = random.choice(self.academic_transitions)
        return f"{t}, {sentence}"

    def _safe_passive_transform(self, sentence):
        if len(sentence.split()) < 4:
            return sentence
        if random.random() < 0.5:
            return sentence + " This can be observed empirically."
        return sentence

    def _replace_with_synonyms_context(self, sentence):
        if not self.nlp:
            return sentence
        try:
            doc = self.nlp(sentence)
            out = []
            for token in doc:
                if token.is_punct or token.is_space:
                    out.append(token.text)
                    continue
                if token.pos_ in ("NOUN", "VERB", "ADJ", "ADV"):
                    lemma = token.lemma_.lower()
                    if lemma in self.academic_word_map and random.random() < 0.6:
                        rep = random.choice(self.academic_word_map[lemma])
                        out.append(rep.capitalize() if token.text.istitle() else rep)
                        continue
                out.append(token.text)
            return " ".join(out)
        except Exception:
            return sentence

    def _final_clean_sentence(self, s):
        s = re.sub(r"\s+", " ", s).strip()
        if s and s[-1] not in ".!?":
            s += "."
        return s

    def _post_process_text(self, t):
        t = re.sub(r"\s+,", ",", t)
        t = re.sub(r"\s+\.", ".", t)
        return t.strip()

# -------------------
# Example usage
# -------------------
if __name__ == "__main__":
    text = "We use this method to make results better. It's important for performance."
    humanizer = AcademicTextHumanizer(p_passive=0.3, p_synonym_replacement=0.6)
    print(humanizer.humanize_text(text, use_passive=True, use_synonyms=True))
