import random
import re

class MicroAcademicHumanizer:
    """Ultra-lightweight text humanizer for Vercel / AWS Lambda."""

    def __init__(
        self,
        p_synonym=0.4,
        p_transition=0.3,
        p_rhythm=0.5,
        seed=None
    ):
        if seed is not None:
            random.seed(seed)

        self.p_synonym = p_synonym
        self.p_transition = p_transition
        self.p_rhythm = p_rhythm

        # Transitions and rhythm patterns
        self.transitions = [
            "Moreover,", "Furthermore,", "Consequently,", "Therefore,",
            "Hence,", "In addition,", "Notably,", "Importantly,"
        ]
        self.rhythm_patterns = [
            lambda s: s + " This highlights a broader implication.",
            lambda s: "To elaborate, " + s[0].lower() + s[1:] if len(s) > 10 else s,
            lambda s: s.replace(",", ";").replace(" and ", ", while "),
            lambda s: s.replace(".", ", and").rstrip(",") + ".",
        ]

        # Tiny synonym dictionary (academic style)
        self.synonyms = {
            "use": ["utilize", "employ", "apply"],
            "make": ["create", "construct", "develop"],
            "good": ["effective", "beneficial", "advantageous"],
            "bad": ["ineffective", "detrimental", "unfavorable"],
            "show": ["demonstrate", "illustrate", "reveal"],
            "help": ["assist", "facilitate", "support"],
            "important": ["crucial", "essential", "vital"],
            "new": ["novel", "innovative", "recent"],
            "big": ["significant", "substantial", "considerable"],
        }

    # ------------------------------
    # Core logic
    # ------------------------------
    def humanize_text(self, text):
        if not text or not isinstance(text, str):
            return "Invalid input"

        # simple sentence split
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        transformed = []

        for sent in sentences:
            if not sent:
                continue
            s = self._expand_contractions(sent)

            if random.random() < self.p_transition:
                s = f"{random.choice(self.transitions)} {s}"

            if random.random() < self.p_synonym:
                s = self._replace_synonyms(s)

            if random.random() < self.p_rhythm:
                s = self._vary_rhythm(s)

            transformed.append(s)

        return " ".join(transformed)

    # ------------------------------
    # Helpers
    # ------------------------------
    def _expand_contractions(self, sentence):
        mapping = {
            "n't": " not", "'re": " are", "'s": " is", "'ll": " will",
            "'ve": " have", "'d": " would", "'m": " am"
        }
        for k, v in mapping.items():
            sentence = sentence.replace(k, v)
        return sentence

    def _replace_synonyms(self, sentence):
        words = sentence.split()
        new_words = []
        for w in words:
            key = w.lower().strip(".,!?")
            if key in self.synonyms and random.random() < 0.6:
                choice = random.choice(self.synonyms[key])
                if w[0].isupper():
                    choice = choice.capitalize()
                new_words.append(choice)
            else:
                new_words.append(w)
        return " ".join(new_words)

    def _vary_rhythm(self, sentence):
        func = random.choice(self.rhythm_patterns)
        try:
            return func(sentence)
        except Exception:
            return sentence


# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    text = (
        "Artificial intelligence is transforming industries. "
        "Researchers use algorithms to enhance efficiency and accuracy in predictive tasks."
    )
    h = MicroAcademicHumanizer()
    print(h.humanize_text(text))
