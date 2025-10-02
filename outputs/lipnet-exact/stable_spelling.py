"""Preserve upstream spelling probabilities; resolve equal scores alphabetically.

The upstream max(set(...), key=P) uses Python's process-dependent set order.
This fixed rule uses no reference text and is shared by both checkpoints.
"""
from engines import UP

def stabilize_spelling(model):
    parent=type(model.spell)
    class StableSpell(parent):
        def correction(self,word):
            return max(sorted(self.candidates(word)),key=self.P)
    model.spell=StableSpell(str(UP/'common/dictionaries/grid.txt'))
    return model
