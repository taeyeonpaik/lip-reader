from types import SimpleNamespace
from training.deeper import Deeper


def result(a,b,ca=.1,cb=.1):
    return {'failures':0,'macro_wer':(a+b)/2,'by_dataset':{'LRS2':{'wer':a,'cer':ca},'LRS3-TED':{'wer':b,'cer':cb}}}


def test_mixed_selection_does_not_hide_regression_in_other_corpus():
    run=Deeper.__new__(Deeper);run.args=SimpleNamespace(balanced_datasets=True)
    baseline=result(.2,.1)
    assert run.qualifies(result(.19,.09),baseline)
    assert not run.qualifies(result(.17,.101),baseline)
    assert not run.qualifies(result(.19,.09,cb=.11),baseline)
    assert run.selection_score(result(.19,.09))==.14
    failed=result(.19,.09);failed['failures']=1
    assert not run.qualifies(failed,baseline)
