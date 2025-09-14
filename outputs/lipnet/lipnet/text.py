"""Character vocabulary, CTC decoders, and corpus-level edit metrics."""

import math

ALPHABET = "abcdefghijklmnopqrstuvwxyz "
BLANK = 0
NUM_CLASSES = len(ALPHABET) + 1


def normalize(text):
    return " ".join(text.lower().split())


def encode(text):
    text = normalize(text)
    if not text:
        raise ValueError("Transcript must contain at least one character")
    unknown = set(text) - set(ALPHABET)
    if unknown:
        raise ValueError(f"Unsupported characters {sorted(unknown)!r}; spell out digits")
    return [ALPHABET.index(c) + 1 for c in text]


def decode(tokens):
    return "".join(ALPHABET[i - 1] for i in tokens if i != BLANK)


def collapse(path):
    """Merge adjacent repeats BEFORE removing blanks: a,blank,a -> aa."""
    result, previous = [], None
    for token in path:
        token = int(token)
        if token != BLANK and token != previous:
            result.append(token)
        previous = token
    return tuple(result)


def greedy_decode(log_probs):
    """Decode a single [time, classes] array/tensor."""
    return decode(collapse(log_probs.argmax(-1).tolist()))


def logadd(*values):
    maximum = max(values)
    if maximum == -math.inf:
        return maximum
    return maximum + math.log(sum(math.exp(x - maximum) for x in values))


def prefix_beam_search(log_probs, beam_width=200):
    """CTC prefix search in log space, without a language model.

    Each prefix tracks probability mass ending in blank and non-blank
    separately; this distinction permits repeated letters across a blank.
    """
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    beams = {(): (0.0, -math.inf)}
    for row in log_probs.tolist():
        next_beams = {}

        def add(prefix, blank=-math.inf, nonblank=-math.inf):
            old_b, old_nb = next_beams.get(prefix, (-math.inf, -math.inf))
            next_beams[prefix] = (logadd(old_b, blank), logadd(old_nb, nonblank))

        for prefix, (pb, pnb) in beams.items():
            total = logadd(pb, pnb)
            add(prefix, blank=total + row[BLANK])
            for token in range(1, len(row)):
                if prefix and token == prefix[-1]:
                    add(prefix, nonblank=pnb + row[token])
                    add(prefix + (token,), nonblank=pb + row[token])
                else:
                    add(prefix + (token,), nonblank=total + row[token])
        beams = dict(sorted(next_beams.items(), key=lambda kv: logadd(*kv[1]),
                            reverse=True)[:beam_width])
    return decode(max(beams, key=lambda p: logadd(*beams[p])))


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


class ErrorRates:
    def __init__(self):
        self.char_errors = self.chars = self.word_errors = self.words = 0
        self.exact = self.sentences = 0

    def update(self, reference, hypothesis):
        reference, hypothesis = normalize(reference), normalize(hypothesis)
        self.char_errors += edit_distance(reference, hypothesis)
        self.chars += len(reference)
        self.word_errors += edit_distance(reference.split(), hypothesis.split())
        self.words += len(reference.split())
        self.exact += reference == hypothesis
        self.sentences += 1

    def compute(self):
        return {"cer": self.char_errors / max(self.chars, 1),
                "wer": self.word_errors / max(self.words, 1),
                "sentence_accuracy": self.exact / max(self.sentences, 1),
                "sentences": self.sentences}
