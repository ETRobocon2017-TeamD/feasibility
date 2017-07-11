import mmap
import os
import sys
import forkTestChild

sharedMemory = mmap.mmap(-1, 13)
sharedMemory.write(b"Hello world!")

pid1 = os.fork()

# 親プロセスが os.fork を呼ぶと、子プロセスのPIDが取得できる
# フォークされた子プロセスは pid1 に 0 を持つ 

if pid1 == 0:  # In a child process
# 
  sharedMemory.seek(0)
  print('Im Child1')
  sys.exit()

pid2 = os.fork()

if pid2 == 0:
  sharedMemory.seek(0)
  print('Im Child2')
  sys.exit()

os.wait()
print('Im parent')
print('Done')
