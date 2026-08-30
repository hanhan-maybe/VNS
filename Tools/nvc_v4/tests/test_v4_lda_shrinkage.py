from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def test_v4_lda_is_shrinkage():
    m=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto")
    assert m.solver=="lsqr" and m.shrinkage=="auto"
