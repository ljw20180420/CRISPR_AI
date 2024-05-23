import datasets
import re

_DESCRIPTION = """\
This is a test data set from sx.
"""

_URLS = {
    "condition": "train.scaffold",
    "alg": "train.alg"
}

# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class sxTestDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features(
                {
                    "count": datasets.Value("int32"),
                    "upEnd": datasets.Value("int32"),
                    "mid": datasets.Value("string"),
                    "downEnd": datasets.Value("int32"),
                    "upRef": datasets.Value("string"),
                    "upCut": datasets.Value("int32"),
                    "downCut": datasets.Value("int32"),
                    "downRef": datasets.Value("string"),
                    "condition": datasets.Value("string"),
                    "query": datasets.Value("string")
                }
            )
        )
    
    def _split_generators(self, dl_manager: datasets.DownloadManager) -> list[datasets.SplitGenerator]:
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                "filepath": downloaded_files
            }),
        ]
    
    def _generate_examples(self, filepath):
        lowfind = re.compile("[ACGTN][acgtn]")
        with open(filepath["alg"]) as alg, open(filepath["condition"]) as cond:
            condSet = cond.readlines()
            for id_, line in enumerate(alg):
                ref = next(alg).replace("-", "")
                ref1End = lowfind.search(ref).span()[1]
                upRef = ref[:ref1End].upper()
                downRef = ref[ref1End:].upper()
                query = next(alg).replace("-", "")
                _, count, _, refId, _, _, _, upEnd, _, mid, downEnd, _, _, _, _, upCut, downCut = line.split("\t")
                count, refId, upEnd, downEnd, upCut, downCut = int(count), int(refId), int(upEnd), int(downEnd), int(upCut), int(downCut)
                yield id_, {
                    "count": count,
                    "upEnd": upEnd,
                    "mid": mid,
                    "downEnd": downEnd - len(upRef),
                    "upRef": upRef,
                    "upCut": upCut,
                    "downCut": downCut - len(upRef),
                    "downRef": downRef,
                    "condition": condSet[refId],
                    "query": query
                }

