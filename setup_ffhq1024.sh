# the first argument to this script should be the path to the ffhq_images1024x1024 folder
# the same path should be provided as the `data_root` argument to train.py
cd $1
mkdir train
mkdir train/0
mkdir valid
mkdir valid/0
for i in $(seq -f "%05g" 0 64999); do
    mv $i.png train/0
done
for i in $(seq -f "%05g" 65000 69999); do
    mv $i.png valid/0
done
