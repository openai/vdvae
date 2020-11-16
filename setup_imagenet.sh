if [ "$1" == "imagenet32" ]; then

echo "downloading imagenet32"
wget http://www.image-net.org/small/train_32x32.tar
wget http://www.image-net.org/small/valid_32x32.tar
tar -xvf train_32x32.tar
tar -xvf valid_32x32.tar
python files_to_npy.py train_32x32/ imagenet32-train.npy
python files_to_npy.py valid_32x32/ imagenet32-valid.npy

elif [ "$1" == "imagenet64" ]; then

echo "downloading imagenet64"
wget http://www.image-net.org/small/train_64x64.tar
wget http://www.image-net.org/small/valid_64x64.tar
tar -xvf train_64x64.tar
tar -xvf valid_64x64.tar
python files_to_npy.py train_64x64/ imagenet64-train.npy
python files_to_npy.py valid_64x64/ imagenet64-valid.npy

else

echo "please pass the string imagenet32 or imagenet64 as an argument"

fi



