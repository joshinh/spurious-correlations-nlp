# encoding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""SNLI CAD"""

from __future__ import absolute_import, division, print_function

import json
import os

import datasets


_CITATION = """\
@misc{gururangan2018annotation,
      title={Annotation Artifacts in Natural Language Inference Data}, 
      author={Suchin Gururangan and Swabha Swayamdipta and Omer Levy and Roy Schwartz and Samuel R. Bowman and Noah A. Smith},
      year={2018},
      eprint={1803.02324},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
TODO Implement
"""


class CadConfig(datasets.BuilderConfig):
    """BuilderConfig for SnliHard."""

    def __init__(self, **kwargs):
        """BuilderConfig for SnliHard.
            Args:
        .
              **kwargs: keyword arguments forwarded to super.
        """
        super(CadConfig, self).__init__(version=datasets.Version("0.9.0", ""), **kwargs)


class Cad(datasets.GeneratorBasedBuilder):
    """SNLI CAD"""

    BUILDER_CONFIGS = [
        CadConfig(
            name="plain_text",
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "premise": datasets.Value("string"),
                    "hypothesis": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=["entailment", "neutral", "contradiction"]),
                }
            ),
            # No default supervised_keys (as we have to pass both premise
            # and hypothesis as input).
            supervised_keys=None,
            homepage="https://www.nyu.edu/projects/bowman/SnliHard/",
            citation=_CITATION,
        )

    def _vocab_text_gen(self, filepath):
        for _, ex in self._generate_examples(filepath):
            yield " ".join([ex["premise"], ex["hypothesis"]])

    def _split_generators(self, dl_manager):

        #downloaded_path = dl_manager.download(
        #    "https://nlp.stanford.edu/projects/snli/snli_1.0_test_hard.jsonl"
        #)
        downloaded_path = "/scratch/nhj4247/robustness/bias-probing/data/cad/manual.tsv"

        return [
            datasets.SplitGenerator(name="test", gen_kwargs={"filepath": downloaded_path}),
        ]

    def _generate_examples(self, filepath):
        """Generate mnli examples.
        Args:
          filepath: a string
        Yields:
          dictionaries containing "premise", "hypothesis" and "label" strings
        """
        
        for idx, line in enumerate(open(filepath, "rb")):
            if idx == 0:
                continue  # skip header
            line = line.strip().decode("utf-8")
            split_line = line.split("\t")
            yield idx, {
                    "premise": split_line[1],
                    "hypothesis": split_line[2],
                    "label": split_line[3]
                }

