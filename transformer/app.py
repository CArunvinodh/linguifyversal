import ssl
import random
import warnings
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Disable warnings for cleaner serverless logs
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------
# Initialize spaCy + NLTK
# ------------------------------
try:
    NLP_GLOBAL = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    NLP_GLOBAL = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

def download_nltk_resources():
    """Ensure required NLTK data is available"""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    resources = [
        'punkt', 'wordnet', 'averaged_perceptron_tagger', 
        'averaged_perceptron_tagger_eng', 'omw-1.4'
    ]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"⚠️ Error downloading {resource}: {str(e)}")

download_nltk_resources()


# ------------------------------
# Main Humanizer Class
# ------------------------------
class AcademicTextHumanizer:
    """
    Lightweight serverless-safe academic text humanizer.
    Produces formal, natural-sounding text suitable for AI detection bypass.
    """

    def __init__(
        self,
        p_passive=0.3,
        p_synonym_replacement=0.5,
        p_academic_transition=0.4,
        seed=None
    ):
        if seed is not None:
            random.seed(seed)

        self.nlp = NLP_GLOBAL
        self.p_passive = p_passive
        self.p_synonym_replacement = p_synonym_replacement
        self.p_academic_transition = p_academic_transition

        # Smooth academic transitions
        self.academic_transitions = [
            "Moreover,", "Furthermore,", "In addition,",
            "Consequently,", "Therefore,", "Hence,",
            "Notably,", "As a result,", "Importantly,",
            "In this context,", "On the other hand,"
        ]

        # Contraction map
        self.contraction_map = {
            "n't": " not", "'re": " are", "'s": " is", "'ll": " will",
            "'ve": " have", "'d": " would", "'m": " am",
        }

    # ------------------------------
    # Core Function
    # ------------------------------
    def humanize_text(self, text, use_passive=False, use_synonyms=False):
        if not text or not isinstance(text, str):
            return "Invalid input text"

        doc = self.nlp(text)
        transformed_sentences = []

        for sent in doc.sents:
            sentence = sent.text.strip()
            if not sentence:
                continue

            # 1️⃣ Expand contractions
            sentence = self.expand_contractions(sentence)

            # 2️⃣ Randomly add academic transitions
            if random.random() < self.p_academic_transition and not sentence.startswith(tuple(self.academic_transitions)):
                sentence = self.add_academic_transition(sentence)

            # 3️⃣ Optionally convert some sentences to passive style
            if use_passive and random.random() < self.p_passive:
                sentence = self.convert_to_passive(sentence)

            # 4️⃣ Optionally replace some words with academic synonyms
            if use_synonyms and random.random() < self.p_synonym_replacement:
                sentence = self.replace_with_synonyms(sentence)

            transformed_sentences.append(sentence)

        return " ".join(transformed_sentences)

    # ------------------------------
    # Contraction Expansion
    # ------------------------------
    def expand_contractions(self, sentence):
        tokens = word_tokenize(sentence)
        expanded_tokens = []
        for token in tokens:
            lower_token = token.lower()
            replaced = False
            for contraction, expansion in self.contraction_map.items():
                if contraction in lower_token and lower_token.endswith(contraction):
                    new_token = lower_token.replace(contraction, expansion)
                    if token[0].isupper():
                        new_token = new_token.capitalize()
                    expanded_tokens.append(new_token)
                    replaced = True
                    break
            if not replaced:
                expanded_tokens.append(token)
        return " ".join(expanded_tokens)

    # ------------------------------
    # Academic Transitions
    # ------------------------------
    def add_academic_transition(self, sentence):
        transition = random.choice(self.academic_transitions)
        return f"{transition} {sentence}"

    # ------------------------------
    # Passive Conversion (simple heuristic)
    # ------------------------------
    def convert_to_passive(self, sentence):
        doc = self.nlp(sentence)
        subj = [t for t in doc if t.dep_ == "nsubj" and t.head.pos_ == "VERB"]
        dobj = [t for t in doc if t.dep_ == "dobj"]
        if subj and dobj:
            subject = subj[0]
            obj = dobj[0]
            verb = subject.head
            passive_form = f"{obj.text} is {verb.lemma_} by {subject.text}"
            return sentence.replace(f"{subject.text} {verb.text} {obj.text}", passive_form)
        # fallback to a natural passive-like extension
        if random.random() < 0.5:
            return sentence + " This is generally acknowledged in academic contexts."
        return sentence

    # ------------------------------
    # Synonym Replacement
    # ------------------------------
    def replace_with_synonyms(self, sentence):
        tokens = word_tokenize(sentence)
        pos_tags = nltk.pos_tag(tokens)
        new_tokens = []

        for (word, pos) in pos_tags:
            clean_word = word.strip('.,!?;:"').lower()
            if pos.startswith(("J", "N", "V", "R")) and len(clean_word) > 3 and wordnet.synsets(clean_word):
                if random.random() < 0.5:
                    synonyms = self._get_simple_synonyms(clean_word, pos)
                    if synonyms:
                        chosen = random.choice(synonyms)
                        if word[0].isupper():
                            chosen = chosen.capitalize()
                        new_tokens.append(chosen)
                        continue
            new_tokens.append(word)
        return " ".join(new_tokens)

    def _get_simple_synonyms(self, word, pos):
        wn_pos = None
        if pos.startswith("J"):
            wn_pos = wordnet.ADJ
        elif pos.startswith("N"):
            wn_pos = wordnet.NOUN
        elif pos.startswith("R"):
            wn_pos = wordnet.ADV
        elif pos.startswith("V"):
            wn_pos = wordnet.VERB

        synonyms = set()
        for syn in wordnet.synsets(word, pos=wn_pos):
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != word.lower() and len(name) > 3:
                    synonyms.add(name)
                    if len(synonyms) >= 5:
                        break
            if len(synonyms) >= 5:
                break
        return list(synonyms)


# ------------------------------
# Example Test Run
# ------------------------------
if __name__ == "__main__":
    text = """Artificial intelligence is transforming industries. 
    Researchers use various algorithms to enhance learning efficiency."""
    humanizer = AcademicTextHumanizer()
    print(humanizer.humanize_text(text, use_passive=True, use_synonyms=True))
