import numpy as np
import pandas as pd

buf = np.zeros([500,4])
with open("nohup.out", "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "[savi]" in line:
            nums = line.split(":")
            for j in range(4):
                buf[i][j]=float(nums[j+1][:12])
DF = pd.DataFrame(buf)
DF.to_csv("data1.csv")
