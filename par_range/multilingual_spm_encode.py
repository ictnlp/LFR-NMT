#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import contextlib
import sys

import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="sentencepiece model to use for encoding"
    )
    parser.add_argument(
        "--inputs", type=str, default='.', help="path to input file"
    )
    parser.add_argument(
        "--outputs", type=str, default='.', help="path to output file"
    )
    parser.add_argument(
        "--lang-file", type=str, default='', help='the language list file'
    )

    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.Load(args.model)

    def encode(l):
        return sp.EncodeAsPieces(l)
    #   key: third column, three chars
    #   value: second colum, __xx_XX__
    lang_list = {}
    f_lang = open(args.lang_file, 'r')
    l = f_lang.readline()
    while l:
        item = l.strip().split('\t')
        if len(item) == 3:
            lang_list[item[2]] = item[1][2:-2]
        l = f_lang.readline()

    for key1 in lang_list.keys():
        for key2 in lang_list.keys():
            if key1 != key2 and (key1 == 'eng' or key2 == 'eng'):
                f1 = open(args.inputs + '/' + key1 + '.dev', 'r')
                f2 = open(args.inputs + '/' + key2 + '.dev', 'r')
                g1 = open(args.outputs + '/' + lang_list[key1] + '-' + lang_list[key2] + '.spm.' + lang_list[key1], 'w')
                g2 = open(args.outputs + '/' + lang_list[key1] + '-' + lang_list[key2] + '.spm.' + lang_list[key2], 'w')
                a = f1.readline()
                b = f2.readline()
                while a:
                    tmp1 = encode(a)
                    tmp2 = encode(b)
                    g1.write(' '.join(tmp1) + '\n')
                    g2.write(' '.join(tmp2) + '\n')
                    a = f1.readline()
                    b = f2.readline()


if __name__ == "__main__":
    main()
