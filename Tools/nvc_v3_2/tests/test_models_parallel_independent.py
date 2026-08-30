from ..development import MODEL_DEFS

def test_models_parallel_independent():
    names = [x[0] for x in MODEL_DEFS]
    assert names == ["B0-primary", "M1_P_SPEC_SHORT", "M2_PE_EUS_STFT_SPARSE", "M3_PE_TF_COMPACT_LR", "M5_PE_TF_COMPACT_SVM"]
    assert all(x[3] is not None for x in MODEL_DEFS)
