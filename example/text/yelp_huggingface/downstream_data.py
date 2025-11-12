from pe.data import Data
from pe.callback import SaveTextToCSV
from pe.constant.data import VARIATION_API_FOLD_ID_COLUMN_NAME

for rp in [2, 10, 50]:
    data = Data()
    data.load_checkpoint(f"/content/drive/MyDrive/SecPE/yelp_pii_diff_secpe600_new_{rp}/checkpoint/000000020")
    data = data.filter({VARIATION_API_FOLD_ID_COLUMN_NAME: -1})
    SaveTextToCSV(output_folder=f"/content/drive/MyDrive/SecPE/yelp_pii_diff_secpe600_new_{rp}/")(data)
