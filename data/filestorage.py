# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import hashlib
import io
import logging
import pathlib
from collections import Counter
from typing import Union, Optional

import click
import lmdb
import tqdm
import networkx as nx


__version__: str = "0.1.0-alpha"
__revision__: int = 2
__author__: str = "Dimitrios Karageorgiou"
__email__: str = "dkarageo@iti.gr"


class LMDBFileStorage:
    """A file storage for handling large datasets based on LMDB."""
    def __init__(self,
                 db_path: pathlib.Path,
                 map_size: int = 1024*1024*1024*1024,  # 1TB
                 read_only: bool = False,
                 max_readers: int = 128):
        self.db: lmdb.Environment = lmdb.open(
            str(db_path),
            map_size=map_size,
            subdir=False,
            readonly=read_only,
            max_readers=max_readers,
            lock=False,
            sync=False
        )

    def open_file(self, file_id: str, mode: str = "r") -> Union[io.TextIOWrapper, io.BytesIO]:
        """Returns a file-like stream of a file in the database."""
        with self.db.begin(buffers=True) as trans:
            data = trans.get(file_id.encode("utf-8"))
        stream: io.BytesIO = io.BytesIO(data)

        if mode == "r":
            reader: io.TextIOWrapper = io.TextIOWrapper(stream)
        elif mode == "b":
            reader: io.BytesIO = stream
        else:
            raise RuntimeError(f"Unsupported file mode: '{mode}'. Only 'r' and 'b' are supported.")

        return reader

    def write_file(self, file_id: str, file_data: bytes) -> None:
        with self.db.begin(write=True) as trans:
            trans.put(file_id.encode("utf-8"), file_data)

    def get_all_ids(self) -> list[str]:
        with self.db.begin() as trans:
            cursor = trans.cursor()
            ids: list[str] = [k for k, _ in cursor]
        return ids

    def close(self) -> None:
        self.db.close()

