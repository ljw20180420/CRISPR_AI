from diffusers import DiffusionPipeline, __version__
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path

class FOREcasTPipeline(DiffusionPipeline):
    def to_json_string(self) -> str:
        """
        Serializes the configuration instance to a JSON string.

        Returns:
            `str`:
                String containing all the attributes that make up the configuration instance in JSON format.
        """
        config_dict = self._internal_dict if hasattr(self, "_internal_dict") else {}
        config_dict["_class_name"] = [self.__module__, self.__class__.__name__]
        config_dict["_diffusers_version"] = __version__

        def to_json_saveable(value):
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, Path):
                value = value.as_posix()
            return value

        config_dict = {k: to_json_saveable(v) for k, v in config_dict.items()}
        # Don't save "_ignore_files" or "_use_default_values"
        config_dict.pop("_ignore_files", None)
        config_dict.pop("_use_default_values", None)

        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def __init__(self, FOREcasT_model, MAX_DEL_SIZE):
        super().__init__()

        self.register_modules(FOREcasT_model=FOREcasT_model)
        self.MAX_DEL_SIZE = MAX_DEL_SIZE
        self.lefts = np.concatenate([
            np.arange(-DEL_SIZE, 1)
            for DEL_SIZE in range(self.MAX_DEL_SIZE, -1, -1)
        ] + [np.zeros(20, np.int64)])
        self.rights = np.concatenate([
            np.arange(0, DEL_SIZE + 1)
            for DEL_SIZE in range(self.MAX_DEL_SIZE, -1, -1)
        ] + [np.zeros(20, np.int64)])
        self.inss = (self.MAX_DEL_SIZE + 2) * (self.MAX_DEL_SIZE + 1) // 2 * [""] + ["A", "C", "G", "T", "AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC", "TG", "TT"]

    def __call__(self, feature):
        assert feature.shape[1] == len(self.lefts), "the possible mutation number of the input feature does not fit the pipeline"
        return {
            "proba": F.softmax(self.FOREcasT_model(feature)["logit"], dim=-1),
            "left": self.lefts,
            "right": self.rights,
            "ins_seq": self.inss
        }