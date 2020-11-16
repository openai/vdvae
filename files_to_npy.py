import sys
import numpy as np
import imageio
import glob
import os

if __name__ == "__main__":
    print("moving images in", sys.argv[1], "to", sys.argv[2])
    files = glob.glob(os.path.join(sys.argv[1], "*.png"))
    shape = imageio.imread(files[0]).shape
    data = np.zeros(shape=(len(files), *shape), dtype=np.uint8)
    for idx, f in enumerate(files):
        data[idx] = imageio.imread(f)
    np.save(sys.argv[2], data)
