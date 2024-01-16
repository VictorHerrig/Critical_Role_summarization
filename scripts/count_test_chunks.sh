grep -f data/test_files <(ls CRD3/data/aligned\ data/c\=4/) | xargs -I{} cat CRD3/data/aligned\ data/c\=4/{} | grep CHUNK -c
