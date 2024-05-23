import datasets
import re
import subprocess

_DESCRIPTION = """\
This is a test data set from sx.
"""

cond = "train.condition"
algs = [
    "train.alg"
]

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
        condition_file = dl_manager.download_and_extract(cond)
        alg_files = dl_manager.download_and_extract(algs)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                "condition_file": condition_file,
                "alg_files": alg_files
            }),
        ]
    
    def _generate_examples(self, condition_file, alg_files):
        lowfind = re.compile("[ACGTN][acgtn]")
        with open(condition_file, "r") as cond:
            condSet = cond.readlines()
            id_ = 0
            for alg_file in alg_files:
                with open(alg_file, "r") as alg:
                    with subprocess.Popen(f'''sed 'N;N;s/\\n/\\t/g' | shuf | awk -F "\t" '{{
                            for (i = 1; i < NF - 2; ++i)
                                printf("%s\\t", $i)
                            for (i = NF - 2; i <= NF; ++i)
                                printf("%s\\n", $i)
                        }}' ''', shell=True, executable="/bin/bash", stdin=alg, stdout=subprocess.PIPE).stdout as algShuf:
                        for line in algShuf:
                            line = line.decode()
                            ref = next(algShuf).decode().replace("-", "")
                            ref1End = lowfind.search(ref).span()[1]
                            upRef = ref[:ref1End].upper()
                            downRef = ref[ref1End:].upper()
                            query = next(algShuf).decode().replace("-", "")
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
                            id_ = id_ + 1

