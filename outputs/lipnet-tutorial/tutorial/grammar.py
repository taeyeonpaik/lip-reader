"""GRID sentence grammar; no reference transcript is accepted by the decoder."""
from functools import lru_cache
import random
import numpy as np
from .accelerated import VOCABULARY,BLANK,greedy_decode

SLOTS = (
 ('Command',('bin','lay','place','set')),
 ('Color',('blue','green','red','white')),
 ('Preposition',('at','by','in','with')),
 ('Letter',tuple(c for c in 'abcdefghijklmnopqrstuvwxyz' if c!='w')),
 ('Number',('zero','one','two','three','four','five','six','seven','eight','nine')),
 ('Ending',('again','now','please','soon')),
)

def generate_sentence(rng=None):
    rng=rng or random.SystemRandom()
    return ' '.join(rng.choice(words) for _,words in SLOTS)

def valid_sentence(text):
    parts=text.lower().split()
    return len(parts)==6 and all(p in words for p,(_,words) in zip(parts,SLOTS))

@lru_cache(maxsize=100000)
def next_characters(prefix):
    parts=prefix.split(' ');stage=len(parts)-1
    if stage>=len(SLOTS) or any(p not in SLOTS[i][1] for i,p in enumerate(parts[:-1])):return (),False
    partial=parts[-1];words=SLOTS[stage][1]
    allowed={w[len(partial)] for w in words if w.startswith(partial) and len(w)>len(partial)}
    if partial in words and stage<5:allowed.add(' ')
    return tuple(sorted(allowed)),stage==5 and partial in words

def constrained_decode(logits,beam=64):
    """Approximate CTC prefix beam search, restricted to the full GRID grammar.

    Each prefix tracks paths ending in blank/nonblank separately so repeated
    letters need intervening blanks. No prompt, transcript or filename is used.
    """
    if hasattr(logits,'detach'):probs=logits.float().log_softmax(-1).detach().cpu().numpy()
    else:
        x=np.asarray(logits,dtype=np.float64);x=x-x.max(-1,keepdims=True);probs=x-np.log(np.exp(x).sum(-1,keepdims=True))
    if probs.ndim!=2 or probs.shape[1]!=41:raise ValueError('Expected [frames,41] logits')
    ids={c:i+1 for i,c in enumerate(VOCABULARY)}
    paths={'':(0.,-np.inf)}
    for row in probs:
        new={}
        def add(prefix,blank=None,nonblank=None):
            a,b=new.get(prefix,(-np.inf,-np.inf))
            new[prefix]=(np.logaddexp(a,blank) if blank is not None else a,np.logaddexp(b,nonblank) if nonblank is not None else b)
        for prefix,(pb,pn) in paths.items():
            total=np.logaddexp(pb,pn);add(prefix,blank=total+row[BLANK])
            if prefix:add(prefix,nonblank=pn+row[ids[prefix[-1]]])
            for char in next_characters(prefix)[0]:
                previous=pb if prefix and char==prefix[-1] else total
                add(prefix+char,nonblank=previous+row[ids[char]])
        paths=dict(sorted(new.items(),key=lambda x:np.logaddexp(*x[1]),reverse=True)[:beam])
    complete=[(p,np.logaddexp(*v)) for p,v in paths.items() if next_characters(p)[1]]
    if not complete:return ''
    return max(complete,key=lambda x:x[1])[0]

def word_error(reference,prediction):
    a=reference.lower().split();b=prediction.lower().split();last=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1):row.append(min(last[j]+1,row[-1]+1,last[j-1]+(x!=y)))
        last=row
    return {'errors':last[-1],'words':len(a),'wer':last[-1]/len(a) if a else 0.,'exact':a==b}
