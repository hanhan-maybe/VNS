import numpy as np
import pandas as pd
from ..validation import _finite_frame

def test_structural_missing_not_imputed():
    d = pd.DataFrame({"teacher_label":["NVC_CORE","PREVOID_PROGRESSIVE"],"subject":["A","B"],"base_eligible":[True,True],"x":[1.0,np.nan]})
    out = _finite_frame(d, ("x",))
    assert out.model_scorable.tolist() == [True, False]
    assert out.loc[1,"model_failure_reason"] == "STRUCTURAL_FEATURE_MISSING"
