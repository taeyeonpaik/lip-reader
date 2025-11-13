"""Corpus error rates. Failed recognition counts as deleting the reference."""
import re


def normalize(text):
    return " ".join(re.sub(r"[^A-Z0-9' ]", " ", text.upper().replace("’", "'")).split())


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(a != b)))
        previous = current
    return previous[-1]


def summarize(rows):
    word_errors = words = character_errors = characters = exact = failures = 0
    for row in rows:
        ref = normalize(row["reference"])
        hyp = normalize(row.get("prediction", "")) if not row.get("error") else ""
        word_errors += edit_distance(ref.split(), hyp.split())
        words += len(ref.split())
        character_errors += edit_distance(ref, hyp)
        characters += len(ref)
        exact += ref == hyp
        failures += bool(row.get("error"))
    return {"clips": len(rows), "word_errors": word_errors, "reference_words": words,
            "wer": word_errors/words if words else None,
            "character_errors": character_errors, "reference_characters": characters,
            "cer": character_errors/characters if characters else None,
            "exact_sentences": exact, "failures": failures}
