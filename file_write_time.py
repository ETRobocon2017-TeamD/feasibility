#!/usr/bin/env python3
u"""EV3でのファイル書き込み速度を調べる
検証元の記事：http://qiita.com/takayoshiotake/items/72c015acd725d35be48a

実行する前に書き込み用ファイルを作っておくこと
$ touch DRIVER
$ python3 file_write_time.py --count=1000
"""
import os
import datetime
from optparse import OptionParser

TEST_COUNT = 1000
DRIVER_FILE = 'DRIVER'
DEVICE_FILE = '/sys/class/tacho-motor/motor0/speed_sp'  # motor0は環境に合わせて存在するデバイスに変更する
COMMAND = b'100'


def test(file_path, file_type, test_count):
    u"""ファイル書き込みのテスト

    Args:
        file_path (str): 書き込むファイル
        file_type (str): ファイル種別（print用）
        test_count (int): 1回の計測で書き込む回数
    """
    fd = os.open(file_path, flags=(os.O_WRONLY | os.O_SYNC))
    start = datetime.datetime.now()
    for i in range(test_count):
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, COMMAND)
    os.close(fd)
    elapsed_time = (datetime.datetime.now() - start).microseconds
    print('{}: {:.4f}us'.format(file_type, elapsed_time / test_count))


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-c', '--count', action='store', type='int', dest='test_count', default=TEST_COUNT,
                      help="1回の計測で書き込む回数")
    options, _ = parser.parse_args()
    test(DRIVER_FILE, 'normal file', options.test_count)
    test(DEVICE_FILE, 'device file', options.test_count)

    # ループ数が少ないと時間がかかる？
    # for count in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
    #     test(DRIVER_FILE, 'normal file', count)
    #     test(DEVICE_FILE, 'device file', count)
