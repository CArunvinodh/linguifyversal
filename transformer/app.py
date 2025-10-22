import json
import random
import re
from http.server import BaseHTTPRequestHandler


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
    
    def humanize_text(self, text):
        """Main method to humanize text."""
        if not text or not isinstance(text, str):
            return "Invalid input"
        
        # Simple sentence split
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
    
    def _expand_contractions(self, sentence):
        """Expand common contractions."""
        mapping = {
            "n't": " not", "'re": " are", "'s": " is", "'ll": " will",
            "'ve": " have", "'d": " would", "'m": " am"
        }
        for k, v in mapping.items():
            sentence = sentence.replace(k, v)
        return sentence
    
    def _replace_synonyms(self, sentence):
        """Replace words with academic synonyms."""
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
        """Vary sentence rhythm for naturalness."""
        func = random.choice(self.rhythm_patterns)
        try:
            return func(sentence)
        except Exception:
            return sentence


class handler(BaseHTTPRequestHandler):
    """Vercel serverless function handler."""
    
    def _set_headers(self, status_code=200):
        """Set common response headers."""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def _send_json_response(self, data, status_code=200):
        """Send JSON response."""
        self._set_headers(status_code)
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self._set_headers(200)
    
    def do_GET(self):
        """Handle GET requests - return API info."""
        response = {
            'message': 'Academic Text Humanizer API',
            'usage': 'Send POST request with JSON body: {"text": "your text here"}',
            'optional_params': {
                'p_synonym': 'Probability of synonym replacement (0-1, default: 0.4)',
                'p_transition': 'Probability of adding transitions (0-1, default: 0.3)',
                'p_rhythm': 'Probability of rhythm variation (0-1, default: 0.5)',
                'seed': 'Random seed for reproducibility (integer, optional)'
            }
        }
        self._send_json_response(response)
    
    def do_POST(self):
        """Handle POST requests - process text humanization."""
        try:
            # Read and parse request body
            content_length = int(self.headers.get('Content-Length', 0))
            
            if content_length == 0:
                self._send_json_response({
                    'error': 'Empty request body',
                    'message': 'Please provide JSON with "text" field'
                }, 400)
                return
            
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Extract parameters
            text = data.get('text', '')
            
            if not text:
                self._send_json_response({
                    'error': 'Missing text parameter',
                    'message': 'Please provide "text" field in JSON body'
                }, 400)
                return
            
            # Optional parameters
            p_synonym = float(data.get('p_synonym', 0.4))
            p_transition = float(data.get('p_transition', 0.3))
            p_rhythm = float(data.get('p_rhythm', 0.5))
            seed = data.get('seed', None)
            
            # Validate probabilities
            if not all(0 <= p <= 1 for p in [p_synonym, p_transition, p_rhythm]):
                self._send_json_response({
                    'error': 'Invalid probability values',
                    'message': 'All probability values must be between 0 and 1'
                }, 400)
                return
            
            # Initialize humanizer with parameters
            humanizer = MicroAcademicHumanizer(
                p_synonym=p_synonym,
                p_transition=p_transition,
                p_rhythm=p_rhythm,
                seed=seed
            )
            
            # Process text
            result = humanizer.humanize_text(text)
            
            # Send successful response
            self._send_json_response({
                'success': True,
                'original': text,
                'humanized': result,
                'parameters': {
                    'p_synonym': p_synonym,
                    'p_transition': p_transition,
                    'p_rhythm': p_rhythm,
                    'seed': seed
                }
            })
            
        except json.JSONDecodeError:
            self._send_json_response({
                'error': 'Invalid JSON',
                'message': 'Request body must be valid JSON'
            }, 400)
            
        except ValueError as e:
            self._send_json_response({
                'error': 'Invalid parameter value',
                'message': str(e)
            }, 400)
            
        except Exception as e:
            self._send_json_response({
                'error': 'Internal server error',
                'message': str(e)
            }, 500)
