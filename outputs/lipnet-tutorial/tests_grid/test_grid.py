import numpy as np,torch
from tutorial.grammar import *
from tutorial.runtime import fixed_input,read_rgb,ROOT

def test_generated_sentences_follow_full_grammar():
    rng=random.Random(1);sentences={generate_sentence(rng) for _ in range(300)}
    assert len(sentences)>280 and all(valid_sentence(s) for s in sentences)
    assert not valid_sentence('bin blue at w two please')

def test_ctc_grammar_recovers_repeated_letters():
    sentence='set green with p three soon';ids={c:i+1 for i,c in enumerate(VOCABULARY)}
    sequence=[]
    for c in sentence:sequence.extend([ids[c],BLANK])
    logits=np.full((len(sequence),41),-15.)
    logits[np.arange(len(sequence)),sequence]=15.
    assert constrained_decode(logits)==sentence

def test_prefix_accepts_only_allowed_positions():
    assert 'w' not in next_characters('bin red at ')[0]
    assert next_characters('bin red at s nine again')[1]
    assert not next_characters('blue bin ')[0]

def test_word_error_counts_insertions():
    assert word_error('bin red','bin blue extra')['wer']==1.


def test_one_take_records_only_first_three_seconds():
    from tutorial.live import OneTakeSession
    sizes=[]
    def process(frames):sizes.append(len(frames));return 'test'
    take=OneTakeSession(process)
    for i in range(150):take.accept(np.zeros((96,96,3),np.uint8),i/25)
    assert take.wait_until_idle(5)
    assert sizes==[75]
    assert take.snapshot()['closed']
