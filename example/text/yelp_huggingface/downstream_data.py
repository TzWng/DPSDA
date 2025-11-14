from pe.data import Data
from pe.callback import SaveTextToCSV
from pe.constant.data import VARIATION_API_FOLD_ID_COLUMN_NAME

# for rp in [2, 10, 50]:
#     data = Data()
#     data.load_checkpoint(f"/content/drive/MyDrive/SecPE/yelp_pii_diff_secpe600_new_{rp}/checkpoint/000000010")
#     data = data.filter({VARIATION_API_FOLD_ID_COLUMN_NAME: -1})
#     SaveTextToCSV(output_folder=f"/content/drive/MyDrive/SecPE/yelp_pii_diff_secpe600_new_{rp}/")(data)

for k in [200, 1000, 1200, 1600, 1800]:
    data = Data()
    data.load_checkpoint(f"/content/drive/MyDrive/SecPE/yelp_different_K/yelp_huggingface_gpt2_secpe_10_{k}_random_5000_20/000000005")
    data = data.filter({VARIATION_API_FOLD_ID_COLUMN_NAME: -1})
    SaveTextToCSV(output_folder=f"/content/drive/MyDrive/SecPE/yelp_different_K/yelp_huggingface_gpt2_secpe_10_{k}_random_5000_20/")(data)


