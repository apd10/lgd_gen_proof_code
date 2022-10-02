import sys
fname = sys.argv[1]
import os
statinfo = os.stat(fname)
print('{},{},{},{}'.format(statinfo.st_size // 10**5 / 10, "MB", statinfo.st_size // 10**2 / 10, "KB"))

