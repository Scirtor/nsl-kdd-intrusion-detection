import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from nslkdd.data import FEATURES, CATEGORICAL, validate_features
from nslkdd.features import FeatureEngineer, build_pipeline


def sample():
    d={c:[0.,1.,2.,3.] for c in FEATURES if c not in CATEGORICAL}
    d.update(protocol_type=['tcp','udp','tcp','udp'],service=['http','dns','http','dns'],flag=['SF']*4)
    return pd.DataFrame(d)


def test_target_and_difficulty_cannot_enter_model():
    x=sample();x['attack_type']=['normal','bad','normal','bad'];x['difficulty']=21;x['label']=[0,1,0,1]
    assert validate_features(x).columns.tolist()==FEATURES
    assert FeatureEngineer().fit_transform(x).shape[1]==43


def test_unknown_categories_are_supported_without_refitting():
    x=sample();pipeline=build_pipeline(LogisticRegression()).fit(x,[0,1,0,1])
    before=pipeline.named_steps['preprocess'].named_transformers_['category'].categories_[1].copy()
    novel=x.copy();novel['service']='unseen_service'
    assert len(pipeline.predict(novel))==len(x)
    np.testing.assert_array_equal(before,pipeline.named_steps['preprocess'].named_transformers_['category'].categories_[1])


def test_preprocessing_uses_training_statistics_only():
    x=sample();pipeline=build_pipeline(LogisticRegression()).fit(x,[0,1,0,1])
    scaler=pipeline.named_steps['preprocess'].named_transformers_['numeric'].named_steps['scale']
    before=scaler.mean_.copy()
    extreme=x.copy();extreme['src_bytes']=1e12
    pipeline.predict(extreme)
    np.testing.assert_array_equal(scaler.mean_,before)


def test_engineering_is_finite_and_does_not_mutate_input():
    x=sample();before=x.copy(deep=True);out=FeatureEngineer().fit_transform(x)
    pd.testing.assert_frame_equal(x,before)
    assert np.isfinite(out['bytes_ratio']).all()
    np.testing.assert_allclose(out.bytes_ratio,x.src_bytes/(x.dst_bytes+1))


@pytest.mark.parametrize('case',['missing','infinite','negative','text','duplicates'])
def test_invalid_inputs_fail_clearly(case):
    x=sample()
    if case=='missing':x=x.drop(columns=['service'])
    if case=='infinite':x['src_bytes']=np.inf
    if case=='negative':x['src_bytes']=-1
    if case=='text':x['src_bytes']='not a number'
    if case=='duplicates':x=pd.concat([x,x[['service']]],axis=1)
    with pytest.raises((ValueError,TypeError)):validate_features(x)
