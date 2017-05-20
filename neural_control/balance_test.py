u"""OpenAI gymのCarPole-v0をQ-Learning（Neural Network版）で学習する"""
import gc
import json
import math
import random
import socket
import time

import ev3dev.ev3 as ev3

from environment import Action
from neuron import NeuralNetwork


class Agent(object):
    u"""エージェント"""

    def __init__(self, network_file_path):
        self.action_value = {}  # Q値
        self.gamma = 0.95  # 割引率γ
        self.epsilon = 0.15  # 探索率ε
        self.network = NeuralNetwork(network_file_path)

    def decide_action(self, state, greedy=False, should_save_output=False):
        """方策に応じて行動を選択する"""
        # ランダムに行動を決定するのにはaction_valuesの値が必要ないが、学習のためにはネットワークを順伝搬させておく必要がある
        action_values = self.network.forward(state, should_save_output=should_save_output)
        if not greedy and random.random() < self.epsilon:
            # ε-greedyアルゴリズムにより、self.epsilonの確率で探索行動を取る
            # 確率的に適当に行動を選択する
            return self._explore()
        else:
            # 最適の行動を選択する(greedy)
            return self._greedy(action_values)

    @staticmethod
    def _explore():
        """ランダムな行動（探索行動）をとる"""
        if random.random() < 0.5:
            return Action.ACTION1
        else:
            return Action.ACTION2

    @staticmethod
    def _greedy(action_values):
        """行動価値が最も高い行動を選択する"""
        max_action_value = float('-inf')
        max_action = None
        for action in Action:
            action_value = action_values[action.value]
            if action_value > max_action_value:
                max_action_value = action_value
                max_action = action
        return max_action


class Robot(object):
    u"""ロボット本体"""

    BASE_SLEEP_TIME = 0.02
    DEG_TO_RAD = (2 * math.pi) / 360
    GYRO_OFFSET = -86
    SERVER_HOST = ('192.168.0.209', 8888)

    def __init__(self):
        self.right_motor = ev3.LargeMotor('outA')
        self.left_motor = ev3.LargeMotor('outC')
        self.gyro_sensor = ev3.GyroSensor('in4')
        self.agent = Agent('network.pickle')
        self.touch_sensor = ev3.TouchSensor('in1')

    def run(self):
        u"""ロボット稼働"""
        try:
            self._main_loop()
        finally:
            self._stop()

    def _main_loop(self):
        u"""ロボットメインループ"""
        self.count_per_rot = self.left_motor.count_per_rot
        sample_list = []

        self._calibration()

        for i_episode in range(100):
            # 起こしてもらえるまで待つ
            ev3.Sound.tone([(400, 30, 100), (400, 30, 0)]).wait()
            tmp_angle, _ = self.gyro_sensor.rate_and_angle
            while abs(tmp_angle - self.GYRO_OFFSET) > 5:
                tmp_angle, _ = self.gyro_sensor.rate_and_angle
                print('wait for stand up...', tmp_angle)
                if self.touch_sensor.value():
                    self._calibration()
                time.sleep(0.5)

            print('ready')
            ev3.Sound.tone([(400, 100, 300), (400, 100, 300), (400, 100, 300), (800, 200, 0)]).wait()
            print('go')

            gc.disable()
            self.left_motor.position = 0
            self.right_motor.position = 0
            training_sample = self._balance()
            sample_list.append(training_sample)
            self._stop()
            gc.enable()

            if i_episode <= 10 or i_episode % 10 == 0:
                # トレーニング結果を送信して学習してもらう
                params = self._train_on_server(sample_list)
                self.agent.network.update_network(params)
                sample_list = []

    def _balance(self):
        inputs_list = []
        action_list = []
        elapsed_times = []
        for _ in range(500):
            start_time = time.time()
            gyro_angle, gyro_rate = self.gyro_sensor.rate_and_angle
            gyro_angle -= self.GYRO_OFFSET

            if abs(gyro_angle) > 20:
                # 倒れた
                ev3.Sound.tone([(800, 800, 0)])
                print('taoreta ', gyro_angle, gyro_rate)
                break
            left_motor_position = 0  # self.left_motor.position
            left_motor_speed = 0  # self.left_motor.speed / self.count_per_rot

            # Neural Network
            inputs = (
                (left_motor_position - gyro_angle) / 500,
                (left_motor_speed - gyro_rate) / 500,
                gyro_angle * self.DEG_TO_RAD,
                gyro_rate * self.DEG_TO_RAD
            )
            inputs_list.append(inputs)
            decided_action = self.agent.decide_action(inputs, greedy=True)
            action_list.append(decided_action)
            if decided_action == Action.ACTION1:
                pwm = -100
            elif decided_action == Action.ACTION2:
                pwm = -70
            elif decided_action == Action.ACTION3:
                pwm = 0
            elif decided_action == Action.ACTION4:
                pwm = 70
            else:
                pwm = 100

            self.right_motor.run_direct(duty_cycle_sp=pwm)
            self.left_motor.run_direct(duty_cycle_sp=pwm)
            # 余った時間はsleep
            elapsed_second = time.time() - start_time
            elapsed_times.append(elapsed_second)
            if elapsed_second < self.BASE_SLEEP_TIME:
                sleep_time = self.BASE_SLEEP_TIME - elapsed_second
                time.sleep(sleep_time)
        print('total')
        for time_, inputs, action in zip(elapsed_times, inputs_list, action_list):
            print('time: {:0.3f} / input: {: .3f}, {: .3f}, {: .3f}, {: .3f} / decide: {}'.
                  format(time_, inputs[0], inputs[1], inputs[2], inputs[3], action.value)
                  )
        return inputs_list

    def _stop(self):
        self.left_motor.stop()
        self.right_motor.stop()

    def _calibration(self):
        # ジャイロのキャリブレーション
        # このときうつ伏せに寝かせていること
        self.gyro_sensor.mode = 'GYRO-CAL'
        self.gyro_sensor.mode = 'GYRO-G&A'
        ev3.Sound.tone([(800, 300, 0)]).wait()

    def _train_on_server(self, data):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect(self.SERVER_HOST)
            # 学習サンプルデータ送信
            data_json = json.dumps(data)
            sock.sendall(bytes(data_json + '\n', 'UTF-8'))
            # 学習後パラメータ受信
            sock_rfile = sock.makefile()
            received_data = sock_rfile.readline().strip()
            print(received_data)
        return json.loads(received_data)

if __name__ == '__main__':
    robot = Robot()
    robot.run()
