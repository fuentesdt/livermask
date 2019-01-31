#!/bin/bash

# Script for segmentation
# Jonas Actor
# 30 Jan 2019
#
# Usage:
# ./makemodel.sh <location/of/dbfile.csv> <numepochs> <kfolds> <location/of/directory/for/output>
#

dbfile=$1
numepochs=$2
kfolds=$3
outdir=$4
let "lastfold = $kfolds - 1"

python3 liver2.py --builddb --dbfile=$dbfile
python3 liver2.py --dbfile=$dbfile --numepochs=$numepochs --kfolds=1 --idfold=0 --trainmodel --outdir=$outdir 
for n in $(seq 0 $lastfold)
	do
		python3 liver2.py --dbfile=$dbfile --numepochs=$numepochs --kfolds=$kfolds --idfold=$n --trainmodel --outdir=$outdir
       	done
python3 liver2.py --dbfile=$dbfile --numepochs=$numepochs --kfolds=$kfolds --setuptestset --outdir=$outdir

