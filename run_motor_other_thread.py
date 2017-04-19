#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
u"""モーターへのrun指示を別スレッドで行うテスト

メインスレッドが4ms以内に動いてくれればとりあえず良いか？
モーター側が詰まってメインスレッドからの速度変更指示がたまったときは、古い指示を捨てれば良いか？

■構成
　メインスレッド：センサーの値を読み取る。モータースレッドに指示を送る
　モータースレッド：メインスレッドからの指示を受けてモーターを回転・停止させる
"""
import datetime
import time
import threading
import queue

import ev3dev.ev3 as ev3


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

    def end_thread(self):
        self._is_loop = False

    def loop(self):
        u"""メインループからの指示を受けるループ"""
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
                self._motor.run_forever(speed_sp=command.speed)
                times.append(datetime.datetime.now())
            elif command.command == MotorCommand.STOP:
                self._motor.stop()
                print('motor_stop')
        self._motor.stop()


class Robot(object):
    u"""ロボット本体"""
    BASE_SLEEP_TIME_US = 20000

    def __init__(self):
        self.motor_right = Motor('outA')
        self.motor_left = Motor('outC')
        self.motor_tail = Motor('outB')
        self.gyro_sensor = None

    def run(self):
        u"""ロボット稼働"""
        try:
            # TODO: 各モーターの待受スレッドを立てる
            motor_left_thread = threading.Thread(target=self.motor_left.loop, name='motor_left_thread')
            motor_right_thread = threading.Thread(target=self.motor_right.loop, name='motor_left_thread')
            motor_tail_thread = threading.Thread(target=self.motor_tail.loop, name='motor_left_thread')
            motor_left_thread.start()
            motor_right_thread.start()
            motor_tail_thread.start()
            self._main_loop()
        except:
            pass
        finally:
            self.stop()

    def stop(self):
        u"""ロボット停止"""
        self.motor_left.end_thread()
        self.motor_right.end_thread()
        self.motor_tail.end_thread()
        self.motor_left.stop()
        self.motor_right.stop()
        self.motor_tail.stop()

    def _main_loop(self):
        u"""ロボットメインループ"""
        delta = 100
        current_speed = 0
        elapsed_times = []
        for _ in range(100):
            start = datetime.datetime.now()
            # TODO: センサーから現在の値を取得
            # TODO: 値に応じて各Motorに指示
            if current_speed >= 1000 or current_speed <= -1000:
                delta *= -1
            current_speed += delta
            self.motor_right.run(speed=current_speed)
            self.motor_left.run(speed=-current_speed)
            # TODO: 4msから経過時間を引いた時間でsleep
            elapsed_microsecond = (datetime.datetime.now() - start).microseconds
            elapsed_times.append(elapsed_microsecond)
            if elapsed_microsecond < self.BASE_SLEEP_TIME_US:
                sleep_time = (self.BASE_SLEEP_TIME_US - elapsed_microsecond) / 1000000
                time.sleep(sleep_time)
        print('\n'.join([str(time_) for time_ in elapsed_times]))


if __name__ == '__main__':
    robot = Robot()
    robot.run()
