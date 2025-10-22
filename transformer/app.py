import ssl
import random
import warnings
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# Setup and Warnings Handling
# -----------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load spaCy model globally
NLP_GLOBAL = spacy.load("en_core_web_sm")

def download_nltk_resources():
    """Ensure NLTK dependencies are present."""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
        ssl._create_default_https_context = _create_unverified_https_context
    except AttributeError:
        pass

    resources = [
        'punkt', 'punkt_tab',
        'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng',
        'wordnet'
    ]
    for r in resources:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass

download_nltk_resources()

# -----------------------------
# Main Class
# -----------------------------
class AcademicTextHumanizer:
    """
    Transforms text into a more formal, human-like academic style.
    Includes contraction expansion, transitions, optional passive voice, and
    synonym-based paraphrasing with semantic similarity.
    """

    def __init__(
        self,
        model_name='paraphrase-MiniLM-L6-v2',
        p_passive=0.25,
        p_synonym_replacement=0.35,
        p_academic_transition=0.35,
        seed=None
    ):
        if seed is not None:
            random.seed(seed)

        self.nlp = NLP_GLOBAL
        self.model = SentenceTransformer(model_name)

        # Transformation probabilities
        self.p_passive = p_passive
        self.p_synonym_replacement = p_synonym_replacement
        self.p_academic_transition = p_academic_transition

        # Transitions for academic coherence
        self.academic_transitions = [
            "Moreover,", "Additionally,", "Furthermore,", "Hence,",
            "Therefore,", "Consequently,", "Nonetheless,", "Nevertheless,"
        ]

    # -----------------------------
    # Core Method
    # -----------------------------
    def humanize_text(self, text, use_passive=False, use_synonyms=False):
        doc = self.nlp(text)
        transformed = []

        for sent in doc.sents:
            s = sent.text.strip()
            if not s:
                continue

            s = self.expand_contractions(s)

            # Randomly insert academic transition
            if random.random() < self.p_academic_transition:
                s = self.add_academic_transitions(s)

            # Optionally apply passive
            if use_passive and random.random() < self.p_passive:
                s = self.convert_to_passive(s)

            # Optionally apply synonym replacement
            if use_synonyms and random.random() < self.p_synonym_replacement:
                s = self.replace_with_synonyms(s)

            transformed.append(s)

        # Combine and cleanup
        result = " ".join(transformed)
        result = result.replace(" ,", ",").replace(" .", ".")
        result = " ".join(result.split())
        return result.strip()

    # -----------------------------
    # Sub-methods
    # -----------------------------
    def expand_contractions(self, sentence):
        contraction_map = {
            "n't": " not", "'re": " are", "'s": " is", "'ll": " will",
            "'ve": " have", "'d": " would", "'m": " am"
        }
        tokens = word_tokenize(sentence)
        expanded = []
        for token in tokens:
            lower = token.lower()
            replaced = False
            for contraction, expansion in contraction_map.items():
                if lower.endswith(contraction):
                    new_token = lower.replace(contraction, expansion)
                    if token[0].isupper():
                        new_token = new_token.capitalize()
                    expanded.append(new_token)
                    replaced = True
                    break
            if not replaced:
                expanded.append(token)
        return " ".join(expanded)

    def add_academic_transitions(self, sentence):
        transition = random.choice(self.academic_transitions)
        return f"{transition} {sentence}"

    def convert_to_passive(self, sentence):
        doc = self.nlp(sentence)
        subj = [t for t in doc if t.dep_ == "nsubj" and t.head.dep_ == "ROOT"]
        dobj = [t for t in doc if t.dep_ == "dobj"]
        if subj and dobj:
            subject = subj[0]
            dobject = dobj[0]
            verb = subject.head
            if subject.i < verb.i < dobject.i:
                passive = f"{dobject.text} {verb.lemma_} by {subject.text}"
                original = " ".join([t.text for t in doc])
                chunk = f"{subject.text} {verb.text} {dobject.text}"
                if chunk in original:
                    return original.replace(chunk, passive)
        return sentence

    def replace_with_synonyms(self, sentence):
        tokens = word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        new_tokens = []

        for word, pos in tagged:
            if pos.startswith(('J', 'N', 'V', 'R')) and wordnet.synsets(word):
                if random.random() < 0.5:
                    synonyms = self._get_synonyms(word, pos)
                    if synonyms:
                        chosen = self._select_closest_synonym(word, synonyms)
                        new_tokens.append(chosen if chosen else word)
                    else:
                        new_tokens.append(word)
                else:
                    new_tokens.append(word)
            else:
                new_tokens.append(word)
        return " ".join(new_tokens)

    def _get_synonyms(self, word, pos):
        wn_pos = (
            wordnet.ADJ if pos.startswith("J") else
            wordnet.NOUN if pos.startswith("N") else
            wordnet.ADV if pos.startswith("R") else
            wordnet.VERB if pos.startswith("V") else None
        )

        synonyms = set()
        for syn in wordnet.synsets(word, pos=wn_pos):
            for lemma in syn.lemmas():
                lemma_name = lemma.name().replace("_", " ")
                if lemma_name.lower() != word.lower():
                    synonyms.add(lemma_name)
        return list(synonyms)

    def _select_closest_synonym(self, original, synonyms):
        if not synonyms:
            return None
        orig_emb = self.model.encode(original, convert_to_tensor=True)
        syn_embs = self.model.encode(synonyms, convert_to_tensor=True)
        scores = util.cos_sim(orig_emb, syn_embs)[0]
        best_idx = scores.argmax().item()
        best_score = scores[best_idx].item()
        return synonyms[best_idx] if best_score >= 0.5 else None


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    sample_text = (
        "We use this method to improve accuracy. "
        "It's important to achieve consistent performance in real experiments."
    )

    humanizer = AcademicTextHumanizer(seed=42)
    result = humanizer.humanize_text(sample_text, use_passive=True, use_synonyms=True)
    print("\n🔹 Humanized Output:\n", result)
