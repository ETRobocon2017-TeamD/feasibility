import os
import time
import logging
from ev3dev.auto import *

logger = logging.getLogger(__name__)

# Function for fast reading from sensor files


def read_device(fd):
    fd.seek(0)
    return int(fd.read().decode().strip())

# Function for fast writing to motor files


def write_device(fd, value):
    fd.truncate(0)
    fd.write(str(int(value)))
    fd.flush()

# ファイルを開いてデータを追記する関数


def write_data_to_file_path(data, file_path):
    fd = open(file_path, 'a')
    fd.write("{}\n".format(data))
    fd.close()

if __name__ == '__main__':
    try:
        # logフォルダの生成
        if not os.path.exists('./log/'):
            os.mkdir('./log/')

        # 日本時間に変更
        os.environ['TZ'] = "JST-9"
        time.tzset()
        log_datetime = time.strftime("%Y%m%d%H%M%S")

        # ログ(csv)のファイルパス
        log_file_path = "./log/log_motor_angle_with_voltage_{}.csv".format(log_datetime)

        # バッテリーセットアップ
        battery = PowerSupply()
        # モーターセットアップ
        left_motor = LargeMotor('outC')
        right_motor = LargeMotor('outB')
        left_motor.reset()
        right_motor.reset()
        left_motor.run_direct()
        right_motor.run_direct()

        # バッテリーファイル（電圧読み取り）オープン
        battery_voltage_devfd = open(battery._path + "/voltage_now", "rb")
        # モーターファイル（角度読み取り）オープン
        motor_encoder_left_devfd = open(left_motor._path + "/position", "rb")
        motor_encoder_right_devfd = open(right_motor._path + "/position", "rb")
        # モーターファイル（デューティーサイクル書き込み）オープン
        motor_duty_cycle_left_devfd = open(left_motor._path + "/duty_cycle_sp", "w")
        motor_duty_cycle_right_devfd = open(right_motor._path + "/duty_cycle_sp", "w")

        # ログ(csv)のヘッダーを書き込み
        log_header = "{}, {}, {}, {}".format(
            "log_time",
            "buttery_voltage",
            "motor_angle_left",
            "motor_angle_right")
        write_data_to_file_path(log_header, log_file_path)

        # 左右モーターの回転開始
        write_device(motor_duty_cycle_left_devfd, 100)
        write_device(motor_duty_cycle_right_devfd, 100)

        # 試験停止フラグ
        stopped = False

        # 毎秒ログを生成（電池がしぬまで）
        while not stopped:
            log_time = time.time() # 現在時刻（秒
            buttery_voltage = read_device(battery_voltage_devfd) #バッテリー電圧(μV)
            motor_angle_left = read_device(motor_encoder_left_devfd)
            motor_angle_right = read_device(motor_encoder_right_devfd)

            log = "{}, {}, {}, {}".format(
                log_time,
                buttery_voltage,
                motor_angle_left,
                motor_angle_right)
            write_data_to_file_path(log, log_file_path)

            time.sleep(5)

    except (Exception, KeyboardInterrupt) as ex:
        logger.exception(ex)

        # モーター止める
        left_motor.stop()
        right_motor.stop()

        # ファイルクローズ
        battery_voltage_devfd.close()
        motor_encoder_left_devfd.close()
        motor_encoder_right_devfd.close()
        motor_duty_cycle_left_devfd.close()
        motor_duty_cycle_right_devfd.close()

        # 試験停止フラグをON
        stopped = True
