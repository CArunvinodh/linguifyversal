import ssl
import random
import warnings
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------
# Safe NLTK setup for serverless
# ------------------------------
def safe_nltk_setup():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)

safe_nltk_setup()

# Load spaCy
try:
    NLP_GLOBAL = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    NLP_GLOBAL = spacy.load("en_core_web_sm", disable=["ner", "textcat"])


# ------------------------------
# Humanizer Class
# ------------------------------
class AcademicTextHumanizer:
    """Serverless-safe academic text humanizer with rhythm variation."""

    def __init__(
        self,
        p_passive=0.3,
        p_synonym_replacement=0.5,
        p_academic_transition=0.4,
        p_rhythm_variation=0.5,
        seed=None,
    ):
        if seed is not None:
            random.seed(seed)

        self.nlp = NLP_GLOBAL
        self.p_passive = p_passive
        self.p_synonym_replacement = p_synonym_replacement
        self.p_academic_transition = p_academic_transition
        self.p_rhythm_variation = p_rhythm_variation

        self.academic_transitions = [
            "Moreover,", "Furthermore,", "In addition,", "Consequently,", "Therefore,",
            "Hence,", "Notably,", "Importantly,", "As a result,", "In this regard,"
        ]

        self.rhythm_patterns = [
            lambda s: s.replace(".", ", and").replace(";", ",") + ".",
            lambda s: s + " Interestingly, this observation persists.",
            lambda s: s.replace(",", ";").replace(" and ", ", while "),
            lambda s: s + " This, in turn, highlights a broader implication.",
            lambda s: "To elaborate, " + s[0].lower() + s[1:] if len(s) > 10 else s,
        ]

        self.contraction_map = {
            "n't": " not", "'re": " are", "'s": " is", "'ll": " will",
            "'ve": " have", "'d": " would", "'m": " am",
        }

    # ------------------------------
    # Core transformation
    # ------------------------------
    def humanize_text(self, text, use_passive=False, use_synonyms=False):
        if not text or not isinstance(text, str):
            return "Invalid input text"

        doc = self.nlp(text)
        transformed = []

        for sent in doc.sents:
            sentence = sent.text.strip()
            if not sentence:
                continue

            sentence = self.expand_contractions(sentence)

            if random.random() < self.p_academic_transition:
                sentence = self.add_academic_transition(sentence)

            if use_passive and random.random() < self.p_passive:
                sentence = self.convert_to_passive(sentence)

            if use_synonyms and random.random() < self.p_synonym_replacement:
                sentence = self.replace_with_synonyms(sentence)

            if random.random() < self.p_rhythm_variation:
                sentence = self.vary_sentence_rhythm(sentence)

            transformed.append(sentence)

        result = " ".join(transformed)
        return self._smooth_join(result)

    # ------------------------------
    # Text transformations
    # ------------------------------
    def expand_contractions(self, sentence):
        tokens = word_tokenize(sentence)
        expanded = []
        for token in tokens:
            lower = token.lower()
            for contr, exp in self.contraction_map.items():
                if lower.endswith(contr):
                    token = lower.replace(contr, exp)
                    break
            expanded.append(token)
        return " ".join(expanded)

    def add_academic_transition(self, sentence):
        return f"{random.choice(self.academic_transitions)} {sentence}"

    def convert_to_passive(self, sentence):
        doc = self.nlp(sentence)
        subj = [t for t in doc if t.dep_ == "nsubj" and t.head.pos_ == "VERB"]
        dobj = [t for t in doc if t.dep_ == "dobj"]
        if subj and dobj:
            s, o, v = subj[0], dobj[0], subj[0].head
            passive = f"{o.text} is {v.lemma_} by {s.text}"
            return sentence.replace(f"{s.text} {v.text} {o.text}", passive)
        return sentence + " This notion is frequently acknowledged in research."

    def replace_with_synonyms(self, sentence):
        tokens = word_tokenize(sentence)
        pos_tags = nltk.pos_tag(tokens)
        new_tokens = []
        for word, pos in pos_tags:
            lw = word.lower()
            if pos.startswith(("J", "N", "V", "R")) and len(lw) > 3 and wordnet.synsets(lw):
                if random.random() < 0.5:
                    synonyms = self._get_synonyms(lw, pos)
                    if synonyms:
                        chosen = random.choice(synonyms)
                        if word[0].isupper():
                            chosen = chosen.capitalize()
                        new_tokens.append(chosen)
                        continue
            new_tokens.append(word)
        return " ".join(new_tokens)

    def _get_synonyms(self, word, pos):
        wn_pos = None
        if pos.startswith("J"):
            wn_pos = wordnet.ADJ
        elif pos.startswith("N"):
            wn_pos = wordnet.NOUN
        elif pos.startswith("R"):
            wn_pos = wordnet.ADV
        elif pos.startswith("V"):
            wn_pos = wordnet.VERB

        syns = set()
        for syn in wordnet.synsets(word, pos=wn_pos):
            for lemma in syn.lemmas():
                n = lemma.name().replace("_", " ")
                if n.lower() != word.lower() and 3 < len(n) < 14:
                    syns.add(n)
                    if len(syns) >= 5:
                        break
            if len(syns) >= 5:
                break
        return list(syns)

    def vary_sentence_rhythm(self, sentence):
        func = random.choice(self.rhythm_patterns)
        try:
            return func(sentence)
        except Exception:
            return sentence

    def _smooth_join(self, text):
        text = text.replace(" .", ".").replace(" ,", ",")
        if random.random() < 0.3:
            text = text.replace(". ", "; ").replace(";", ",")
        return text


# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    sample = """Artificial intelligence is transforming industries.
    Researchers use algorithms to enhance efficiency and accuracy in predictive tasks."""
    humanizer = AcademicTextHumanizer()
    print(humanizer.humanize_text(sample, use_passive=True, use_synonyms=True))
