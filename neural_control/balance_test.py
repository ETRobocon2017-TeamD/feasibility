u"""OpenAI gymのCarPole-v0をQ-Learning（Neural Network版）で学習する"""
import enum
import gc
import time
import pickle
import random

import ev3dev.ev3 as ev3


def get_reward(observation):
    """環境の値から報酬を計算する"""
    x, _, angle, _ = observation
    loss = abs(x + 1) ** 2 + abs(angle * 5 + 1) ** 2
    return -loss


class NeuralNetwork(object):
    """ニューラルネットワークの学習管理クラス"""
    INPUT_LAYER_NEURONS = 4  # 入力層ニューロン数
    HIDDEN_LAYER_NEURONS = 16  # 隠れ層ニューロン数
    OUTPUT_LAYER_NEURONS = 2  # 出力層ニューロン数 = Action数

    LEARNING_RATE = 1e-3  # 学習率

    def __init__(self, network_file_path):
        # とりあえずバイアス項はなし
        with open(network_file_path, 'rb') as file:
            self.params = pickle.load(file)
        self.output = None

    def forward(self, x_input, should_save_output=False):
        """ネットワークを順伝搬させて出力を計算する

        入力層のニューロンは順に1, 2, ..., i, ...
        隠れ層と出力層も同様にj, kと添字をふることにする
        ここではuは入力値と重みの総和、φは任意の活性化関数、yはニューロンの出力とする
        φ_hは隠れ層の活性化関数。ここではReLUを使う
        φ_oは隠れ層の活性化関数。ここでは恒等関数を使う
        """
        # 隠れ層の計算
        # u_j = Σ_i { x_i * w_ij }
        u_hidden = self._poor_dot(x_input, self.params['W_INPUT'])
        # y_j = φ_h(u_j)
        y_hidden = self.relu(u_hidden)  # 活性化関数はReLU

        # 出力層の計算
        # u_k = Σ_j { y_j * w_jk }
        u_output = self._poor_dot(y_hidden, self.params['W_HIDDEN'])
        # y_k = φ_o(u_k)
        y_output = u_output  # 活性化関数は恒等関数

        # 誤差逆伝搬で使う出力値
        if should_save_output:
            self.output = {
                'u_hidden': u_hidden,
                'y_hidden': y_hidden,
                'y_output': y_output,
            }
        return y_output

    def back_propagation(self, x_input, target):
        """誤差逆伝搬でネットワークの重みを更新する

        誤差関数Eは、出力が連続値であるため自乗平均をとる
        targetは教師信号の値
        E = Σ_k{ (target_k - y_k)^2 } / 2

        隠れ層 - 出力層間の重みは次の式で更新する
        ηは学習率とする
        w_jk = w_jk - η * Δw_jk
        Δw_jk = ∂E/∂w_jk
              = ∂E/∂y_k * ∂y_k/∂u_k * ∂u_k/∂w_jk
              = (y_k - target_k) * φ_o'(u_k) * y_j
        ここで
        δ_output_k = (y_k - target_k) * φ_o'(u_k)
        とおいておく

        入力層 - 隠れ層間の重みは
        w_ij = w_ij - η * Δw_ij
        Δw_ij = ∂E/∂w_ij
              = Σ_k{ ∂E/∂y_k * ∂y_k/∂u_k * ∂u_k/∂y_j * ∂y_j/∂u_j * ∂u_j/∂x_i }
              = Σ_k{ (y_k - target_k) * φ_h'(u_k) * w_jk * φ'(u_j) * x_i }
              = Σ_k{ δ_output_k * w_jk * φ_h'(u_j) * x_i }
              = Σ_k{ δ_output_k * w_jk } * φ_h'(u_j) * x_i
        """
        if self.output is None:
            return
        # 誤差逆伝搬では順伝搬で計算したニューロン出力値を使う
        u_hidden = self.output['u_hidden']
        y_hidden = self.output['y_hidden']
        y_output = self.output['y_output']

        # 隠れ層 - 出力層間の重みを更新
        # 出力層の活性化関数は恒等関数なので、φ_o'(u_k) = 1
        delta_o = []
        for y_output_k, target_k in zip(y_output, target):
            delta_o.append(y_output_k - target_k)

        for j, y_hidden_j in enumerate(y_hidden):
            for k, delta_o_k in enumerate(delta_o):
                self.params['W_HIDDEN'][j][k] += -self.LEARNING_RATE * y_hidden_j * delta_o_k

        # 入力層 - 隠れ層間の重みを更新
        # φ_h'(u_j)はReLUの微分
        delta_relu = [value > 0 for value in u_hidden]
        # delta_w1_tmpは　Σ_k{ δ_output_k * w_jk } * φ_h'(u_j) までの計算
        delta_w1_dot = self._poor_dot(delta_o, self.params['W_HIDDEN'])
        delta_w1_tmp = []
        for delta_relu_j, delta_w1_dot_j in zip(delta_relu, delta_w1_dot):
            delta_w1_tmp.append(delta_relu_j * delta_w1_dot_j)

        for i, x_input_i in enumerate(x_input):
            for j, delta_w1_j in enumerate(delta_w1_tmp):
                self.params['W_INPUT'][i][j] += -self.LEARNING_RATE * x_input_i * delta_w1_j

    @staticmethod
    def relu(inputs):
        """活性化関数ReLU"""
        return [value if value > 0 else 0 for value in inputs]

    @staticmethod
    def _poor_dot(value_1d, value_2d):
        u"""np.dotの代用。1次元配列と2次元配列のみ受け付ける"""
        outputs = [0] * len(value_2d[0])
        for input_, weight_i in zip(value_1d, value_2d):
            for j, weight in enumerate(weight_i):
                outputs[j] += input_ * weight
        return outputs


class Action(enum.Enum):
    u"""エージェントが取りうる行動"""
    u"""前に全力"""
    ACTION1 = 0

    u"""後ろに全力"""
    ACTION2 = 1


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

    def update_action_value(self, state, decided_action, reward, next_state):
        """Q-Learningアルゴリズムでネットワークを更新する

        元のQ-Learningの更新式は
        Q(St, At) ← (1 - η)Q(St, At) + η(Rt+1 + γ * max_a{Q(St+1, At+1))}
        または式変形して
        Q(St, At) ← η( Rt+1 + γ * max_a{Q(St+1, At+1)} - Q(St, At) )

        この
        Rt+1 + γ * max_a{Q(St+1, At+1)}
        が教師信号targetとなる
        """
        next_action_values = self.network.forward(next_state, should_save_output=False)
        next_max_action_value = max(next_action_values)
        target = list(self.network.output['y_output'])
        # 実際に選択した行動については報酬から教師信号を計算する
        # 　→選択しなかった行動は更新させない
        target[decided_action.value] = reward + self.gamma * next_max_action_value
        # 誤差逆伝搬でネットワークを更新する
        self.network.back_propagation(state, target)

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

    def __init__(self):
        self.right_motor = ev3.LargeMotor('outA')
        self.left_motor = ev3.LargeMotor('outC')
        self.gyro_sensor = ev3.GyroSensor('in4')
        self.agent = Agent('network.pickle')

    def run(self):
        u"""ロボット稼働"""
        try:
            self._main_loop()
        finally:
            self._stop()

    def _main_loop(self):
        u"""ロボットメインループ"""
        elapsed_times = []
        input_list = []
        self.left_motor.position = 0
        self.right_motor.position = 0
        self.gyro_sensor.mode = 'GYRO-G&A'
        gyro_offset = self.gyro_sensor.angle
        count_per_rot = self.left_motor.count_per_rot
        print('ready')
        ev3.Sound.tone([(400, 100, 300), (400, 100, 300), (400, 100, 300), (800, 200, 0)]).wait()
        print('go')
        for _ in range(500):
            start_time = time.time()
            gyro_angle, gyro_rate = self.gyro_sensor.rate_and_angle
            gyro_angle -= gyro_offset

            if abs(gyro_angle) > 45:
                # 倒れた
                ev3.Sound.tone([(800, 800, 0)])
                print('taoreta ', gyro_angle, gyro_rate)
                break
            left_motor_position = self.left_motor.position
            left_motor_speed = self.left_motor.speed / count_per_rot

            # Neural Network
            inputs = (left_motor_position / 500, left_motor_speed / 500, gyro_angle / 100, gyro_rate / 100)
            decided_action = self.agent.decide_action(inputs, greedy=True)
            input_list.append([inputs, decided_action])
            if decided_action == Action.ACTION1:
                pwm = -100
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
        for time_, inputs in zip(elapsed_times, input_list):
            print(time_, inputs)

    def _stop(self):
        self.left_motor.stop()
        self.right_motor.stop()

if __name__ == '__main__':
    gc.disable()
    robot = Robot()
    robot.run()
