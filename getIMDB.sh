#!/bin/sh

OS=$1

if [$OS == "OSX"]
then
    brew install wget
fi

URL=http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
wget $URL
tar -xvzf *.tar.gz

DATADIR=$PWD/../data/
rm aclImdb_v1.tar.gz
mv aclImdb $DATADIR
