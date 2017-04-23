#!/usr/bin/env python3
u"""EV3でのファイル書き込み速度を調べる
検証元の記事：http://qiita.com/takayoshiotake/items/72c015acd725d35be48a

実行する前に書き込み用ファイルを作っておくこと
$ touch DRIVER
"""
import os
import datetime

TEST_COUNT = 1000
DRIVER_FILE = 'DRIVER'
DEVICE_FILE = '/sys/class/tacho-motor/motor0/speed_sp'  # motor0は環境に合わせて存在するデバイスに変更する
COMMAND = b'100'


def test(file_path, file_type):
    fd = os.open(file_path, flags=(os.O_WRONLY | os.O_SYNC))
    start = datetime.datetime.now()
    for i in range(TEST_COUNT):
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, COMMAND)
    os.close(fd)
    elapsed_time = (datetime.datetime.now() - start).microseconds
    print('{}: {}us'.format(file_type, elapsed_time / TEST_COUNT))


if __name__ == '__main__':
    test(DRIVER_FILE, 'normal file')
    test(DEVICE_FILE, 'device file')
