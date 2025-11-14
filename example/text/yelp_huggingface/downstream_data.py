from pe.data import Data
from pe.callback import SaveTextToCSV
from pe.constant.data import VARIATION_API_FOLD_ID_COLUMN_NAME

# --- numpy 2.x 兼容旧 pickle 的别名补丁（放在文件最上方） ---
import sys, numpy as np, numpy.core as npcore
sys.modules['numpy._core'] = npcore
sys.modules['numpy._core.numeric'] = npcore.numeric
# 可选：有时还会需要下面两个
sys.modules.setdefault('numpy._core.multiarray', npcore.multiarray)
sys.modules.setdefault('numpy._core.overrides',  npcore.overrides)

import pandas as pd  # 现在再导入 pandas


# for rp in [2, 10, 50]:
#     data = Data()
#     data.load_checkpoint(f"/content/drive/MyDrive/SecPE/yelp_pii_diff_secpe600_new_{rp}/checkpoint/000000010")
#     data = data.filter({VARIATION_API_FOLD_ID_COLUMN_NAME: -1})
#     SaveTextToCSV(output_folder=f"/content/drive/MyDrive/SecPE/yelp_pii_diff_secpe600_new_{rp}/")(data)

for k in [200, 1000, 1200, 1600, 1800]:
    data = Data()
    data.load_checkpoint(f"/content/drive/MyDrive/SecPE/Yelp_different_K/yelp_huggingface_gpt2_secpe_10_{k}_random_5000_20/000000020")
    data = data.filter({VARIATION_API_FOLD_ID_COLUMN_NAME: -1})
    SaveTextToCSV(output_folder=f"/content/drive/MyDrive/SecPE/Yelp_different_K/yelp_huggingface_gpt2_secpe_10_{k}_random_5000_20/")(data)


