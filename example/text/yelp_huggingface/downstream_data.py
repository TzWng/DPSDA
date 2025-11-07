from pe.data import Data
from pe.callback import SaveTextToCSV
from pe.constant.data import VARIATION_API_FOLD_ID_COLUMN_NAME

data = Data()
data.load_checkpoint("/content/drive/MyDrive/SecPE/yelp_pii_diff_secpe600_10_new/checkpoint/000000005")
data = data.filter({VARIATION_API_FOLD_ID_COLUMN_NAME: -1})
SaveTextToCSV(output_folder="/content/drive/MyDrive/SecPE/yelp_pii_diff_secpe600_10_new/")(data)
