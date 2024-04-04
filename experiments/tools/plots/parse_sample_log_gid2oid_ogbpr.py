

# ogbpr 16 ..._mtest
prefix_sum = [0, 152094, 307866, 457285, 612970, 766751, 920640, 1076199, 1233313, 1382786, 1532218, 1681479, 1838472, 1987760, 2144387, 2293716]

# lids  = [153337, 153312, 153468, 215951, 231053, 240889, 250456, 250472, 285649, 285653] # ogbpr metis ids
original_ids = [  59838] # 319100

lookups = original_ids
found = [ False ] * 10

import numpy as np

for i in range(1):
    f  = f"/data/ds_pre/ogbpr/chunking_16_mtest/p{i}.oid.bin"
    lid2oid = np.fromfile(f, dtype=np.int64)

    print(lid2oid[509])
    exit(0)
    for idx, v in enumerate(lookups):
#        if found[idx]:
#            continue

        for iii in range(lid2oid.size):
            if lid2oid[iii] == v:
#                found[idx] = True
                print(f"orig: {v:06d} <- part:{i:02d} lid:{iii:08d} lid':{(prefix_sum[i]+iii):08d}")
#                break

"""
orig: 057935 <- part:06 lid:00014129 lid':00934769
orig: 758791 <- part:06 lid:00123086 lid':01043726
orig: 839219 <- part:07 lid:00060903 lid':01137102
orig: 085320 <- part:07 lid:00015502 lid':01091701
orig: 557590 <- part:07 lid:00102403 lid':01178602
orig: 858599 <- part:08 lid:00097703 lid':01331016
orig: 2314678 <- part:09 lid:00144812 lid':01527598
orig: 1695554 <- part:10 lid:00081446 lid':01613664
orig: 802029 <- part:10 lid:00115835 lid':01648053
orig: 914457 <- part:10 lid:00027806 lid':01560024
------ reorder
orig: 2314678 <- part:09 lid:00144812 lid':01527598
orig: 057935 <- part:06 lid:00014129 lid':00934769
orig: 858599 <- part:08 lid:00097703 lid':01331016
orig: 1695554 <- part:10 lid:00081446 lid':01613664
orig: 839219 <- part:07 lid:00060903 lid':01137102
orig: 802029 <- part:10 lid:00115835 lid':01648053
orig: 914457 <- part:10 lid:00027806 lid':01560024
orig: 085320 <- part:07 lid:00015502 lid':01091701
orig: 557590 <- part:07 lid:00102403 lid':01178602
orig: 758791 <- part:06 lid:00123086 lid':01043726
"""

original_ids = [2314678,   57935,  858599, 1695554,  839219,  802029,  914457, 85320,  557590,  758791]
