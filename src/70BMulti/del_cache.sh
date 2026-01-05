#!/bin/bash


target_directory="/data02/henry/lmcache_disk/"

cd "$target_directory" || exit 1

rm *

echo "all caches in lmcache_disk are deleted."