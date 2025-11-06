import numpy as np

# from pe.dp import Gaussian
from pe.data import Data
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.logging import execution_logger
from pe.embedding import Embedding
from tqdm import tqdm, trange
from pe.dp.secretdp import Gaussian
import pandas as pd
import re
import time
import pickle

class PE(object):
    """The class that runs the PE algorithm."""

    def __init__(self, mix_data, secrets, population, histogram, embedding, 
                 dp_mode = "gdp", loggers=[], callbacks=[]):
        """Constructor.

        :param priv_data: The private data
        :type priv_data: :py:class:`pe.data.Data`
        :param population: The population algorithm
        :type population: :py:class:`pe.population.Population`
        :param histogram: The histogram algorithm
        :type histogram: :py:class:`pe.histogram.Histogram`
        :param dp: The DP algorithm, defaults to None, in which case the Gaussian mechanism
            :py:class:`pe.dp.Gaussian` is used
        :type dp: :py:class:`pe.dp.DP`, optional
        :param loggers: The list of loggers, defaults to []
        :type loggers: list[:py:class:`pe.logger.Logger`], optional
        :param callbacks: The list of callbacks, defaults to []
        :type callbacks: list[Callable or :py:class:`pe.callback.Callback`], optional
        """
        super().__init__()
        self._mix_data = mix_data
        self._secrets = secrets
        self._population = population
        self._histogram = histogram
        self._embedding = embedding
        self._dp_mode = dp_mode
        self._loggers = loggers
        self._callbacks = callbacks
        self._batch_num = 1000

    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint.

        :param checkpoint_path: The path to the checkpoint
        :type checkpoint_path: str
        :return: The synthetic data
        :rtype: :py:class:`pe.data.Data` or None
        """
        syn_data = Data()
        if not syn_data.load_checkpoint(checkpoint_path):
            return None
        return syn_data

    def _log_metrics(self, syn_data):
        """Log metrics.

        :param syn_data: The synthetic data
        :type syn_data: :py:class:`pe.data.Data`
        """
        if not self._callbacks:
            return
        metric_items = []
        for callback in self._callbacks:
            metric_items.extend(callback(syn_data) or [])
        for logger in self._loggers:
            logger.log(iteration=syn_data.metadata.iteration, metric_items=metric_items)
        for metric_item in metric_items:
            metric_item.clean_up()

    def _get_num_samples_per_label_id(self, num_samples, fraction_per_label_id):
        """Get the number of samples per label id given the total number of samples

        :param num_samples: The total number of samples
        :type num_samples: int
        :param fraction_per_label_id: The fraction of samples for each label id. The fraction does not have to be
            normalized. When it is None, the fraction is assumed to be the same as the fraction of label ids in the
            private data. Defaults to None
        :type fraction_per_label_id: list[float], optional
        :raises ValueError: If the length of fraction_per_label_id is not the same as the number of labels
        :raises ValueError: If the number of samples is so small that the number of samples for some label ids is zero
        :return: The number of samples per label id
        :rtype: np.ndarray
        """
        if fraction_per_label_id is None:
            execution_logger.warning(
                "fraction_per_label_id is not provided. Assuming the fraction of label ids in private data is public "
                "information."
            )
            fraction_per_label_id = self._mix_data.data_frame[LABEL_ID_COLUMN_NAME].value_counts().to_dict()
            fraction_per_label_id = [
                0 if i not in fraction_per_label_id else fraction_per_label_id[i]
                for i in range(len(self._mix_data.metadata.label_info))
            ]
        if len(fraction_per_label_id) != len(self._mix_data.metadata.label_info):
            raise ValueError("fraction_per_label_id should have the same length as the number of labels.")
        fraction_per_label_id = np.array(fraction_per_label_id)
        fraction_per_label_id = fraction_per_label_id / np.sum(fraction_per_label_id)

        target_num_samples_per_label_id = fraction_per_label_id * num_samples
        num_samples_per_label_id = np.floor(target_num_samples_per_label_id).astype(int)
        num_samples_left = num_samples - np.sum(num_samples_per_label_id)
        ids = np.argsort(target_num_samples_per_label_id - num_samples_per_label_id)[::-1]
        num_samples_per_label_id[ids[:num_samples_left]] += 1
        assert np.sum(num_samples_per_label_id) == num_samples
        if np.any(num_samples_per_label_id == 0):
            raise ValueError("num_samples is so small that the number of samples for some label ids is zero.")
        return num_samples_per_label_id

    def _clean_up_loggers(self):
        """Clean up loggers."""
        for logger in self._loggers:
            logger.clean_up()

    def evaluate(self, checkpoint_path):
        """Evaluate the synthetic data.

        :param checkpoint_path: The path to the checkpoint
        :type checkpoint_path: str
        """
        syn_data = self.load_checkpoint(checkpoint_path)
        execution_logger.info(f"Loaded checkpoint from {checkpoint_path}, iteration={syn_data.metadata.iteration}")
        self._log_metrics(syn_data)

    def _split_by_secret(self, mix_data, secret_matrix = None):
        """Secret detection.

        :param mix_data: The mixed data
        :type mix_data: :py:class:`pe.data.Data`
        :return: The public data, private data, and secret matrix
        :rtype: tuple[:py:class:`pe.data.Data`, :py:class:`pe.data.Data`, np.ndarray]
        """
        if secret_matrix is None:
            patterns = [re.compile(t) for t in self._secrets]
            texts = mix_data.data_frame["PE.TEXT"].fillna("").astype(str)

            rows = []
            for text in texts:
                rows.append([1 if p.search(text) else 0 for p in patterns])

            secret_matrix = np.array(rows, dtype=np.int8)

        pub_num = np.sum(np.all(secret_matrix == 0, axis=1))
        sec_nem = secret_matrix.shape[0] - pub_num
        execution_logger.info(f"Secret detection done. Public data: {pub_num}, Private data: {sec_nem}")

        mask_pub = (secret_matrix.sum(axis=1) == 0)
        secret_matrix = secret_matrix[~mask_pub]

        return secret_matrix

    def _split_for_sample(self, mix_data, p, r, secret_matrix=None):
        """Secret detection.

        :param mix_data: The mixed data
        :type mix_data: :py:class:`pe.data.Data`
        :return: The public data, private data, and secret matrix
        :rtype: tuple[:py:class:`pe.data.Data`, :py:class:`pe.data.Data`, np.ndarray]
        """
        if secret_matrix is None:
            patterns = [re.compile(t) for t in self._secrets]
            texts = mix_data.data_frame["PE.TEXT"].fillna("").astype(str)

            rows = []
            for text in texts:
                rows.append([1 if p.search(text) else 0 for p in patterns])

            secret_matrix = np.array(rows, dtype=np.int8)

        mask_pub = (secret_matrix.sum(axis=1) == 0)

        if np.all(mask_pub):
            execution_logger.info(
                f"[resample] All public: {mask_pub.size} rows; no private rows."
                )
            return mix_data
        
        sp = Gaussian(mode = self._dp_mode)
        df_pub = mix_data.data_frame.loc[mask_pub].reset_index(drop=True)
        secret_matrix = secret_matrix[~mask_pub]
        prob = sp.sampling_prob(sec_matrix=secret_matrix, p=p, r=r, num_samples=self._batch_num)
            
        rng = np.random.default_rng(seed=42)
        df_priv = mix_data.data_frame.loc[~mask_pub].reset_index(drop=True)
        df_priv = df_priv.loc[rng.random(len(df_priv)) < prob]

        df_combined = (
        pd.concat([df_pub, df_priv], axis=0).reset_index(drop=True)[mix_data.data_frame.columns]
        )

        mix_data.data_frame = df_combined

        return mix_data
    
    

    def run(
        self,
        num_samples_schedule,
        p, r, label_matrix=None,
        noise_multiplier=None,
        checkpoint_path=None,
        save_checkpoint=True,
        fraction_per_label_id=None,
    ):
        """Run the PE algorithm.

        :param num_samples_schedule: The schedule of the number of samples for each PE iteration. The first element is
            the number of samples for the initial data, and the rest are the number of samples for each PE iteration.
            So the length of the list is the number of PE iterations plus one
        :type num_samples_schedule: list[int]
        :param delta: The delta value of DP
        :type delta: float
        :param epsilon: The epsilon value of DP, defaults to None
        :type epsilon: float, optional
        :param noise_multiplier: The noise multiplier of the DP mechanism, defaults to None
        :type noise_multiplier: float, optional
        :param checkpoint_path: The path to load and save the checkpoint, defaults to None
        :type checkpoint_path: str, optional
        :param save_checkpoint: Whether to save the checkpoint, defaults to True
        :type save_checkpoint: bool, optional
        :param fraction_per_label_id: The fraction of samples for each label id. The fraction does not have to be
            normalized. When it is None, the fraction is assumed to be the same as the fraction of label ids in the
            private data. Defaults to None
        :type fraction_per_label_id: list[float], optional
        :return: The synthetic data
        :rtype: :py:class:`pe.data.Data`
        """
        try:
        #     # Set privacy budget.
        #     self._dp.set_epsilon_and_delta(
        #         num_iterations=len(num_samples_schedule) - 1,
        #         epsilon=epsilon,
        #         delta=delta,
        #         noise_multiplier=noise_multiplier,
        #     )

            label_data = {}
            sp = Gaussian(mode = self._dp_mode)
            for label_id in range(len(self._mix_data.metadata.label_info)):
                execution_logger.info(f"Label {label_id}")
                if np.sum(r) == 0:
                    label_data[label_id] = {
                        "noise": 0
                    }
                else: 
                    sub_mix_data = self._mix_data.filter_label_id(label_id=label_id)
                    secret_matrix = label_matrix[label_id] if label_matrix is not None else None

                    sub_secret_matrix = self._split_by_secret(sub_mix_data, secret_matrix)

                    if sub_secret_matrix is None:
                        sub_sigma = 0
                        label_data[label_id] = {
                            "noise": sub_sigma
                        }
                    else:
                        sub_prob = sp.sampling_prob(sec_matrix=sub_secret_matrix, p=p, r=r, num_samples=self._batch_num)
                        sub_sigma = sp.get_noise_multiplier(
                            p=p, r=r, sec_matrix=sub_secret_matrix,
                            w = sub_prob, num_steps=len(num_samples_schedule)-1, 
                            min_noise_multiplier=1e-1, max_noise_multiplier=500)
                    
                        label_data[label_id] = {
                            "noise": sub_sigma
                        }
                    execution_logger.info(f"noise for mu GDP: {sub_sigma}")

            # Generate or load initial data.
            if checkpoint_path is not None and (syn_data := self.load_checkpoint(checkpoint_path)):
                execution_logger.info(
                    f"Loaded checkpoint from {checkpoint_path}, iteration={syn_data.metadata.iteration}"
                )
            else:
                num_samples_per_label_id = self._get_num_samples_per_label_id(
                    num_samples=num_samples_schedule[0],
                    fraction_per_label_id=fraction_per_label_id,
                )
                syn_data_list = []
                for label_id, label_info in enumerate(tqdm(self._mix_data.metadata.label_info)):
                    syn_data = self._population.initial(
                        label_info=label_info,
                        num_samples=num_samples_per_label_id[label_id],
                    )
                    syn_data.set_label_id(label_id)
                    syn_data_list.append(syn_data)
                syn_data = Data.concat(syn_data_list, metadata=self._mix_data.metadata)
                syn_data.data_frame.reset_index(drop=True, inplace=True)
                syn_data.metadata.iteration = 0
                self._log_metrics(syn_data)


            total_duration = 0.0 

            # Run PE iterations.
            for iteration in trange(syn_data.metadata.iteration + 1, len(num_samples_schedule)):
                execution_logger.info(f"PE iteration {iteration}")
                num_samples_per_label_id = self._get_num_samples_per_label_id(
                    num_samples=num_samples_schedule[iteration],
                    fraction_per_label_id=fraction_per_label_id,
                )
                syn_data_list = []
                # priv_data_list = []

                # Generate synthetic data for each label.
                for label_id in range(len(self._mix_data.metadata.label_info)):
                    execution_logger.info(f"Label {label_id}")
                    sub_mix_data = self._mix_data.filter_label_id(label_id=label_id)
                    secret_matrix = label_matrix[label_id] if label_matrix is not None else None

                    sub_syn_data = syn_data.filter_label_id(label_id=label_id)
                    pack = label_data[label_id]
                    noise = pack["noise"]

                    if np.sum(r) != 0:
                        sub_mix_data = self._split_for_sample(sub_mix_data, p, r, secret_matrix)
                        
                    execution_logger.info(f"noise for mu GDP: {noise}")

                    # DP NN histogram.
                    start_time = time.time()
                    sub_mix_data, sub_syn_data, time_1, time_2 = self._histogram.compute_histogram(
                        priv_data=sub_mix_data, 
                        syn_data=sub_syn_data, 
                    )
                    end_time = time.time()
                    duration = end_time - start_time
                    total_duration += duration

                    # priv_data_list.append(sub_mix_data)
                    sub_syn_data = sp.add_noise(syn_data=sub_syn_data, noise_multiplier=noise)

                    # Generate next population.
                    sub_syn_data = self._population.next(
                        syn_data=sub_syn_data,
                        num_samples=num_samples_per_label_id[label_id],
                    )
                    sub_syn_data.set_label_id(label_id)
                    syn_data_list.append(sub_syn_data)

                syn_data = Data.concat(syn_data_list)
                syn_data.data_frame.reset_index(drop=True, inplace=True)
                syn_data.metadata.iteration = iteration

                # new_mix_data = Data.concat(priv_data_list)
                # self._mix_data = self._mix_data.merge(new_mix_data)

                if save_checkpoint:
                    syn_data.save_checkpoint(checkpoint_path)
                self._log_metrics(syn_data)
        finally:
            self._clean_up_loggers()

        execution_logger.info(f"total computation time: {total_duration}")
        return syn_data
