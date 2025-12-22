from pathlib import Path
import json,time,torch
from training.mwer import MWERExperiment,parser,expected_word_error,restore
args=parser().parse_args(['--run','work/mwer/smoke'])
e=MWERExperiment(args);checks=[]
optimizer=torch.optim.AdamW(e.params,lr=1e-5)
for dataset in ['LRS2','LRS3-TED']:
 row=next(r for r in e.groups['train'] if r['dataset']==dataset)
 tick=time.monotonic();n=e.hypotheses(row);x=e.features(row)
 candidates=n['candidates'];seq=[c['tokens'] for c in candidates]
 scores=e.scores(x,seq)
 with torch.no_grad():
  individual=torch.stack([e.scores(x,[s])[0] for s in seq])
 assert torch.allclose(scores.detach(),individual,atol=2e-4,rtol=1e-4)
 beam=torch.tensor([c['beam_score'] for c in candidates]);delta=(scores.detach()-beam).abs().max().item()
 assert torch.allclose(scores.detach(),beam,atol=2e-3,rtol=1e-4),(scores,beam)
 errors=torch.tensor([c['word_errors'] for c in candidates],dtype=scores.dtype)
 ce=e.conventional_loss(x,row['text']);risk=expected_word_error(scores,errors);loss=risk+.1*ce
 optimizer.zero_grad(set_to_none=True);loss.backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in e.params)
 norm=torch.nn.utils.clip_grad_norm_(e.params,1.,error_if_nonfinite=True);optimizer.step()
 assert any(not torch.equal(t,e.original[k]) for k,t in e.state().items())
 checks.append({'dataset':dataset,'candidate_count':len(seq),'beam_score_max_abs_delta':delta,'batched_individual_scores_match':True,'finite_gradients':True,'loss':float(loss.detach()),'seconds':time.monotonic()-tick})
 restore(e.model,e.original)
report={'state':'passed','trainable_parameters':e.config['trainable_parameters'],'checks':checks,'base_weights_preserved':True}
Path('outputs/lipreading-studio/validation/mwer-feasibility.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
