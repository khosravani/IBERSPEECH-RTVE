#!/bin/bash
# Copyright 2017   Abbas Khosravani
# Apache 2.0.
#
# See README.txt for more info on data required.

set -e

data_root=/home/abbas/abbas/workspace/iberspeech
data_dir=../data/sre16/v2/data/
wav_ref=audiolist

./make_wav.pl $data_root dev1 $wav_ref $data_dir/iberspeech_dev1
./make_wav.pl $data_root dev2 $wav_ref $data_dir/iberspeech_dev2
./make_wav.pl $data_root test $wav_ref $data_dir/iberspeech_test

