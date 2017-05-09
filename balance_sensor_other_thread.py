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
            while not self.command_queue.empty():
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


class BalanceParam(object):
    u"""balanceの入力パラメータ取得クラス
    loopにてひたすら最新値を取得しメモリ上に保管
    使用側はget_paramの戻り値にて取得
    """
    def __init__(self, rightMortor, leftMortor):
        self.right_motor = rightMortor
        self.left_motor = leftMortor
        self.gyro_sensor = ev3.GyroSensor('in4')
        self.battery = ev3.PowerSupply()
        self._is_loop = True
        
        # 最新取得値
        self.gyro_sensor_rate = 0
        self.left_motor_position = 0
        self.right_motor_position = 0
        self.battery_voltage = 0 #mV
    
        self.lock = threading.Lock()

    def get_param(self):
        u"""最新入力パラメータ取得"""
        with self.lock:
            rate = self.gyro_sensor_rate
            lpos = self.left_motor_position
            rpos = self.right_motor_position
            voltage = self.battery_voltage
        
        return rate, lpos, rpos, voltage
    
    def end_thread(self):
        self._is_loop = False
        

    def loop(self):
        u"""デバイスから現在の値を取得"""
        while self._is_loop:
            with self.lock:
                self.gyro_sensor_rate = self.gyro_sensor.rate
            with self.lock:
                self.left_motor_position = self.left_motor.get_position()
            with self.lock:
                self.right_motor_position = self.right_motor.get_position()
            with self.lock:
                self.battery_voltage = self.battery.measured_voltage / 1000 #measured_voltageはマイクロボルトなのでミリボルトにする
            # 適当に1ms sleep
            time.sleep(0.001)


class Robot(object):
    u"""ロボット本体"""
    BASE_SLEEP_TIME_US = balance.EXEC_PERIOD * 1000000

    def __init__(self):
        self.right_motor = Motor('outA')
        self.left_motor = Motor('outC')
        self.tail_motor = Motor('outB')
        self.balance_param = BalanceParam(self.right_motor, self.left_motor)
        # self.gyro_sensor = ev3.GyroSensor('in4')
        # self.battery = ev3.PowerSupply()

    def run(self):
        u"""ロボット稼働"""
        try:
            left_motor_thread = threading.Thread(target=self.left_motor.loop, name='left_motor_thread')
            right_motor_thread = threading.Thread(target=self.right_motor.loop, name='right_motor_thread')
            tail_motor_thread = threading.Thread(target=self.tail_motor.loop, name='tail_motor_thread')
            balance_param_thread = threading.Thread(target=self.balance_param.loop, name='balance_param_thread')
            left_motor_thread.start()
            right_motor_thread.start()
            tail_motor_thread.start()
            balance_param_thread.start()
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
        self.balance_param.end_thread()
        self.left_motor.stop()
        self.right_motor.stop()
        self.tail_motor.stop()

    def _main_loop(self):
        u"""ロボットメインループ"""
        elapsed_times = []
        balance.balance_init()
        print('ready')
        # ジャイロセンサーの値
        # http://python-ev3dev.readthedocs.io/en/latest/sensors.html#ev3dev.core.GyroSensor.rate
        # 電圧(μV)
        # http://python-ev3dev.readthedocs.io/en/latest/other.html#ev3dev.core.PowerSupply.measured_voltage
        # "motor count"（エンコーダ値）
        # XXX: "count" "encode"でAPIドキュメントを探してこれが一番それっぽかったけど合ってるのか、あまり自信なし
        # http://python-ev3dev.readthedocs.io/en/latest/motors.html#ev3dev.core.Motor.position
        for _ in range(100):
            start = datetime.datetime.now()
            # パラメータ取得
            rate, lpos, rpos, voltage = self.balance_param.get_param()
            
            left_pwm, right_pwm = balance.balance_control(
                0,  # forward -100～100, 0で停止
                0,  # turn -100～100, 0で直進
                rate,  # balance.cのecrobot_get_gyro_sensor(NXT_PORT_S4)のつもり
                0,  # offset（角速度）は0固定（起動時は角速度が変化しないように固定しておくこと）
                lpos,  # balance.cのnxt_motor_get_count(NXT_PORT_C)のつもり
                rpos,
                voltage # 
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
