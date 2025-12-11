import glob
import json
import os
import unittest

import pytest


@pytest.mark.nonci
class TestPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.data = []

        for file in glob.glob(os.path.join("localworkspace", "results", "*.jsonl")):
            with open(file, "r") as jf:
                for line in jf:
                    if len(line.strip()) > 0:
                        self.data.append(json.loads(line))
                        print(self.data[-1])

    def test_edgar(self) -> None:
        self.assertTrue(any("King of the English" in line["text"] for line in self.data))

    def test_ambig(self) -> None:
        self.assertTrue(any("Apples and Bananas" in line["text"] for line in self.data))

    def test_dolma(self) -> None:
        self.assertTrue(any("We extensively document Dolma" in line["text"] for line in self.data))
