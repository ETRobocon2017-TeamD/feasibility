#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
u"""balance.cを移植したコードで動かしてみるテスト"""
import datetime
import time
import threading
import queue

import ev3dev.ev3 as ev3

import balance.balance as balance


class MotorCommand(object):
    u"""モーターのコマンド"""
    RUN = 1
    STOP = 2

    def __init__(self, command, speed=0):
        self.command = command
        self.speed = speed


class Motor(object):
    u"""モーター"""

    def __init__(self, address):
        self.command_queue = queue.Queue()
        self._motor = ev3.LargeMotor(address)
        self._is_loop = True

    def run(self, speed):
        u"""モーターを動かす

        Args:
            speed (int): モーターのスピード（1050まで？負の値で逆回転？）
        """
        self.command_queue.put(MotorCommand(MotorCommand.RUN, speed=speed))

    def stop(self):
        u"""モーターを停止する"""
        self.command_queue.put(MotorCommand(MotorCommand.STOP))

    def get_position(self):
        return self._motor.position

    def end_thread(self):
        self._is_loop = False

    def loop(self):
        u"""メインループからの指示を受けるループ"""
        self._motor.position = 0  # balance.cのnxt_motor_set_count(NXT_PORT_C, 0)のつもり
        times = []
        while self._is_loop:
            # メインスレッドからの指示を受信
            command = self.command_queue.get()
            while not self.command_queue.empty() and False:
                try:
                    # 2つ以上queueにあれば古いのを捨てて新しい指示を実行する
                    command = self.command_queue.get_nowait()
                except queue.Empty:
                    # 取れなくても無視する
                    pass

            if command.command == MotorCommand.RUN:
                self._motor.run_direct(duty_cycle_sp=command.speed)
                times.append(datetime.datetime.now())
            elif command.command == MotorCommand.STOP:
                self._motor.stop()
                print('motor_stop')
        self._motor.stop()


class Robot(object):
    u"""ロボット本体"""
    BASE_SLEEP_TIME_US = balance.EXEC_PERIOD * 1000000

    def __init__(self):
        self.right_motor = Motor('outA')
        self.left_motor = Motor('outC')
        self.tail_motor = Motor('outB')
        self.gyro_sensor = ev3.GyroSensor('in4')
        self.battery = ev3.PowerSupply()

    def run(self):
        u"""ロボット稼働"""
        try:
            left_motor_thread = threading.Thread(target=self.left_motor.loop, name='left_motor_thread')
            right_motor_thread = threading.Thread(target=self.right_motor.loop, name='left_motor_thread')
            tail_motor_thread = threading.Thread(target=self.tail_motor.loop, name='left_motor_thread')
            left_motor_thread.start()
            right_motor_thread.start()
            tail_motor_thread.start()
            self._main_loop()
        except Exception as error:
            print(error)
        finally:
            self.stop()

    def stop(self):
        u"""ロボット停止"""
        self.left_motor.end_thread()
        self.right_motor.end_thread()
        self.tail_motor.end_thread()
        self.left_motor.stop()
        self.right_motor.stop()
        self.tail_motor.stop()

    def _main_loop(self):
        u"""ロボットメインループ"""
        elapsed_times = []
        balance.balance_init()
        gyro_offset = self.gyro_sensor.angle  # XXX: 起動時のジャイロの値で良い？
        print('ready')
        # ジャイロセンサーの値
        # XXX: 幾つかモードがあるけどANGでいいんだろうか
        # http://python-ev3dev.readthedocs.io/en/latest/sensors.html#ev3dev.core.GyroSensor.angle
        # 電圧(μV)　これは合ってるはず
        # http://python-ev3dev.readthedocs.io/en/latest/other.html#ev3dev.core.PowerSupply.measured_voltage
        # "motor count"（エンコーダ値）
        # XXX: "count" "encode"でAPIドキュメントを探してこれが一番それっぽかったけど合ってるのか、あまり自信なし
        # http://python-ev3dev.readthedocs.io/en/latest/motors.html#ev3dev.core.Motor.position
        for _ in range(1000):
            start = datetime.datetime.now()
            left_pwm, right_pwm = balance.balance_control(
                0,  # forward -100～100, 0で停止
                0,  # turn -100～100, 0で直進
                self.gyro_sensor.angle,  # balance.cのecrobot_get_gyro_sensor(NXT_PORT_S4)のつもり
                gyro_offset,
                self.left_motor.get_position(),  # balance.cのnxt_motor_get_count(NXT_PORT_C)のつもり
                self.right_motor.get_position(),
                self.battery.measured_voltage / 1000  # measured_voltageはマイクロボルトなのでミリボルトにする
            )
            # balance_controlからは-100～100までのPWM値が返ってくる
            self.right_motor.run(speed=right_pwm)
            self.left_motor.run(speed=left_pwm)

            # 余った時間はsleep
            elapsed_microsecond = (datetime.datetime.now() - start).microseconds
            elapsed_times.append(elapsed_microsecond)
            if elapsed_microsecond < self.BASE_SLEEP_TIME_US:
                sleep_time = (self.BASE_SLEEP_TIME_US - elapsed_microsecond) / 1000000
                time.sleep(sleep_time)
        print('\n'.join([str(time_) for time_ in elapsed_times]))


if __name__ == '__main__':
    robot = Robot()
    robot.run()
