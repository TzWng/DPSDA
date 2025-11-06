"""
This example follows the experimental settings of the GPT-2 Yelp experiments in the ICML 2024 Spotlight paper,
"Differentially Private Synthetic Data via Foundation Model APIs 2: Text" (https://arxiv.org/abs/2403.01749).

The ``model_name_or_path`` parameter can be set to other models on HuggingFace. Note that we use the FastChat
library (https://github.com/lm-sys/FastChat) to manage the conversation template. If the conversation template of your
desired model is not available in FastChat, please register the conversation template in the FastChat library. See the
following link for an example:
https://github.com/microsoft/DPSDA/blob/main/pe/llm/huggingface/register_fastchat/gpt2.py

The saved CSV files contain both the text selected by the histogram and the generated variations of the selected text,
while the original paper (https://arxiv.org/abs/2403.01749) only use the text selected by the histogram for downstream
evaluation. We can extract the desired text from the saved checkpoints; please see
https://microsoft.github.io/DPSDA/getting_started/examples.html#checkpoint-operation
for more details.

For detailed information about parameters and APIs, please consult the documentation of the Private Evolution library:
https://microsoft.github.io/DPSDA/.
"""

from pe.data.text import Yelp
from pe.data import Data
from pe.logging import setup_logging
from pe.runner import PE, SECPE
from pe.population import PEPopulation
from pe.api.text import LLMAugPE
from pe.llm import HuggingfaceLLM
from pe.embedding.text import SentenceTransformer
from pe.histogram import NearestNeighbors
from pe.callback import SaveCheckpoints
from pe.callback import ComputeFID
from pe.callback import SaveTextToCSV
from pe.logger import CSVPrint
from pe.logger import LogPrint
from pe.constant.data import VARIATION_API_FOLD_ID_COLUMN_NAME

import pandas as pd
import os
import numpy as np
import pickle

pd.options.mode.copy_on_write = True


if __name__ == "__main__":
    exp_folder = "/content/drive/MyDrive/SecPE/yelp_pii_diff"
    current_folder = os.path.dirname(os.path.abspath(__file__))

    setup_logging(log_file=os.path.join(exp_folder, "log.txt"))

    data = Yelp(root_dir="/content/drive/MyDrive/SecPE/upload")
    # secrets = ['phillie', 'btwn', 'shh', 'bespoke', 'clot', 'allegria', 'rava', 'repost', 'gracing', 'gaminess', 'sri', 'plateau', 'perms', 'bobas', 'wince', 'virago', 'pana', '15k', 'sunglass', 'translating', 'banal', 'lupe', 'borgata', 'irreplaceable', 'conceptually', 'raided', 'purge', 'toki', 'wollensky', 'encyclopedic', 'surpassing', 'sharpen', 'bayalage', 'sneakily', '204', 'crunchwrap', 'sprinted', 'mythos', 'compromises', 'backdrops', 'dalton', 'umpteenth', 'bei', 'unknowing', 'eves', '2hours', 'gigante', 'marino', 'paccheri', 'micha', 'filomena', 'uploading', 'raffles', 'vikings', 'lindenwood', 'temperamental', 'murphys', 'slainte', 'undeserving', 'olfactory', 'yona', 'orthotics', 'sprinklers', '168', 'untrustworthy', 'fudgy', 'reconfirm', 'verduras', 'depict', 'plesant', 'carmelitas', 'cortadito', 'erie', 'yeesh', 'dipa', 'grump', 'woodworking', '2sp', 'fritatta', 'modem', 'narnia', 'gameday', 'diablitos', 'citadel', 'fc', 'snickered', 'caf', 'tapestry', 'agnes', 'anomalies', 'tolerating', 'cobra', 'gulping', 'outages', 'nobles', 'presenter', 'flopping', 'cupid', 'tabbouli', 'gateau', 'cocoon', 'marries', 'commonwealth', 'gobbling', 'folsom', 'healthiness', 'liquored', 'sniffed', 'shoves', 'résistance', 'garret', 'greene', 'fasciitis', 'northerners', 'characterize', 'abysmally', 'probability', 'distressing', 'smarmy', 'whistling', 'jude', 'enterprises', 'coaxing', 'poser', 'brookie', 'evils', 'discusses', 'calabria', 'waaayyy', 'chasse', 'automobiles', 'instructs', 'bandaids', 'dabs', 'cystic', 'surcharges', 'guthrie', 'wring', 'tills', 'khing', 'ticketmaster', 'dorsett', 'anthonys', 'pyt', 'morally', 'ravaged', 'hte', 'dossant', 'radiohead', 'nuthin', 'crispies', 'tumbling', 'innings', 'intangibles', 'reinforces', 'runnier', 'amazinggggg', 'unconscionable', 'kohlrabi', 'playgrounds', 'fiddlehead', 'quetzally', 'etoile', 'vetted', 'schemes', 'roving', 'cheung', '133', 'hammonton', 'grasping', 'mailers', 'slumped', 'miserly', 'woodlands', 'cemetary', 'lardo', 'cite', 'pivotal', 'était', 'hiro', 'rockfish', 'wonderfull', 'hushpuppy', 'scalded', 'winspear', 'wearable', 'chihuly', 'coaxed', 'pouting', '4hrs', 'horde', 'niçoise', 'capitalist', 'buuuut', 'babylon', 'quotations', 'kickboxing', 'bullion', 'hott', 'dissect', 'bitchin', 'tequenos', 'marti', 'indulgences', 'emotionless', 'chelsey', 'camouflage', 'soggier', 'dodgers', 'teaming', 'coronary', 'fwot', 'admonished', 'crappier', 'isc', 'swimwear', 'singularly', 'totalled', 'marisco', 'inquisitive', 'carissa', 'lampshade', 'tallow', 'segunda', 'retaliation', 'updos', 'handprints', '445', 'johanna', 'pesticides', 'badgering', 'instinctively', 'dekes', 'reigning', 'jeyuk', 'throng', 'krēm', 'miner', 'interracial', 'qualifications', 'curators', 'reproduction', 'chemex', 'chex', 'foodservice', 'mazes', 'polaris', 'wrigley', 'asthmatic', 'downturn', 'brunchy', 'culinarily', 'membrane', 'hopsmith', 'bitched', 'awash', 'jennie', 'spawn', 'hammers', 'veterinarian', 'rotator', 'acceptably', '422', 'epi', 'boswell', 'unfocused', 'mouthy', 'quatro', 'psychologist', 'rani', 'burton', 'unbothered', 'kingfisher', 'eladio', 'expletive', 'bozo', 'laces', 'curacao', 'expereince', 'opp', 'omelete', 'malcolm', 'amour', 'volvo', 'freckles', 'crosswalk', 'uph', 'ofc', 'aunties', 'egress', 'upsale', '134', 'zodiac', 'mimics', '35min', 'preventive', 'bonita']
    secrets = ['ACCOUNTNAME', 'AGE', 'AMOUNT', 'BS', 'BUILDINGNUMBER', 'CITY', 'COMPANYNAME', 'COUNTY', 'CREDITCARDCVV', 'DATE', 'DOB', 'EMAIL', 'FIRSTNAME', 'GENDER', 'HEIGHT', 'IP', 'JOBAREA', 'JOBTITLE', 'JOBTYPE', 'LASTNAME', 'MAC', 'MIDDLENAME', 'NEARBYGPSCOORDINATE', 'PASSWORD', 'PHONENUMBER', 'PIN', 'SECONDARYADDRESS', 'SEX', 'SSN', 'STATE', 'STREET', 'TIME', 'USERAGENT', 'USERNAME', 'VEHICLEVRM', 'ZIPCODE']
    
    llm = HuggingfaceLLM(max_completion_tokens=64, model_name_or_path="gpt2", temperature=1.4)
    api = LLMAugPE(
        llm=llm,
        random_api_prompt_file=os.path.join(current_folder, "random_api_prompt.json"),
        variation_api_prompt_file=os.path.join(current_folder, "variation_api_prompt.json"),
    )
    embedding = SentenceTransformer(model="stsb-roberta-base-v2")
    histogram = NearestNeighbors(
        embedding=embedding,
        mode="cos_sim",
        lookahead_degree=0,
        num_clusters=600,
    )
    population = PEPopulation(
        api=api, initial_variation_api_fold=6, next_variation_api_fold=6, keep_selected=True, selection_mode="rank"
    )

    save_checkpoints = SaveCheckpoints(os.path.join(exp_folder, "checkpoint"))
    # compute_fid = ComputeFID(
    #     priv_data=data, embedding=embedding, filter_criterion={VARIATION_API_FOLD_ID_COLUMN_NAME: -1}
    # )
    # with open("/content/drive/MyDrive/SecPE/compute_fid_yelp.pkl", "wb") as f:
    #     pickle.dump(compute_fid, f)
    with open("/content/drive/MyDrive/SecPE/compute_fid_yelp.pkl", "rb") as f:
        compute_fid = pickle.load(f)
        
    save_text_to_csv = SaveTextToCSV(output_folder=os.path.join(exp_folder, "synthetic_text"))

    csv_print = CSVPrint(output_folder=exp_folder)
    log_print = LogPrint()

    num_private_samples = len(data.data_frame)
    delta = 1.0 / num_private_samples / np.log(num_private_samples)

    pe_runner = SECPE(
        mix_data=data,
        embedding=embedding,
        secrets=secrets,
        population=population,
        histogram=histogram,
        callbacks=[save_checkpoints, save_text_to_csv, compute_fid],
        loggers=[csv_print, log_print],
    )
    J = len(secrets)
    p = np.full(J, 1e-4, dtype=np.float64)
    r = np.array([0.32 , 1.664, 1.384, 0.768, 1.048, 1.944, 1.272, 1.776, 0.824,
                  1.216, 1.104, 0.656, 2.   , 1.328, 1.16 , 0.992, 1.44 , 0.376,
                  1.552, 1.608, 0.488, 1.496, 0.264, 0.096, 0.6  , 0.544, 0.936,
                  0.712, 0.152, 1.888, 1.832, 1.72 , 0.04 , 0.88 , 0.208, 0.432]) * p

    pe_runner.run(
        num_samples_schedule=[5000] * 2,
        p=p, r=r,
        checkpoint_path=os.path.join(exp_folder, "checkpoint"),
    )
