
savepath="data"

python scripts/download.py --save_path $savepath

cat ./part_* > e5_Flat.index
