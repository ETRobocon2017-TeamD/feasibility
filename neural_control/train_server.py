import socketserver
import json
import pickle

import numpy as np

from environment import Action
from environment import get_reward
from smart_neuron import NeuralNetworkNumPy as NeuralNetwork


class Agent(object):
    u"""エージェント"""

    def __init__(self, network_file_path):
        self.action_value = {}  # Q値
        self.gamma = 0.95  # 割引率γ
        self.epsilon = 0.3  # 探索率ε
        self.network = NeuralNetwork(network_file_path)

    def decide_action(self, state, greedy=False, should_save_output=False):
        """方策に応じて行動を選択する"""
        # ランダムに行動を決定するのにはaction_valuesの値が必要ないが、学習のためにはネットワークを順伝搬させておく必要がある
        action_values = self.network.forward(state, should_save_output=should_save_output)
        if not greedy and np.random.rand() < self.epsilon:
            # ε-greedyアルゴリズムにより、self.epsilonの確率で探索行動を取る
            # 確率的に適当に行動を選択する
            return self._explore()
        else:
            # 最適の行動を選択する(greedy)
            return self._greedy(action_values)

    def stock_action_value_delta(self, state, decided_action, reward, next_state):
        """Q-Learningアルゴリズムでネットワークの更新量を計算する

        元のQ-Learningの更新式は
        Q(St, At) ← (1 - η)Q(St, At) + η(Rt+1 + γ * max_a{Q(St+1, At+1))}
        または式変形して
        Q(St, At) ← η( Rt+1 + γ * max_a{Q(St+1, At+1)} - Q(St, At) )

        この
        Rt+1 + γ * max_a{Q(St+1, At+1)}
        が教師信号targetとなる
        """
        next_action_values = self.network.forward(next_state, should_save_output=False)
        next_max_action_value = np.max(next_action_values)
        target = np.array(self.network.output['y_output'])
        # 実際に選択した行動については報酬から教師信号を計算する
        # 　→選択しなかった行動は更新させない
        target[decided_action.value] = reward + self.gamma * next_max_action_value
        # 誤差逆伝搬でネットワークを更新する
        self.network.back_propagation(state, target)

    def update_action_value(self):
        """学習した行動価値関数をネットワークに反映する"""
        self.network.params['W_INPUT'] += self.network.params_delta['W_INPUT']
        self.network.params['W_HIDDEN'] += self.network.params_delta['W_HIDDEN']

    @staticmethod
    def _explore():
        """ランダムな行動（探索行動）をとる"""
        return np.random.choice(Action)

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


class TrainServer(socketserver.StreamRequestHandler):
    def handle(self):
        received_data = self.rfile.readline().strip()
        print(received_data)
        training_samples = json.loads(str(received_data, 'UTF-8'))

        agent = Agent('network.pickle')
        for sample in training_samples:
            for inputs, next_inputs in zip(sample, sample[1:]):
                action = agent.decide_action(inputs, should_save_output=True)
                reward = get_reward(inputs)
                agent.stock_action_value_delta(inputs, action, reward, next_inputs)
        agent.update_action_value()
        params = agent.network.params
        params['W_INPUT'] = params['W_INPUT'].tolist()
        params['W_HIDDEN'] = params['W_HIDDEN'].tolist()
        with open('network.pickle', 'wb') as file:
            pickle.dump(params, file)
        data_json = json.dumps(params)
        self.wfile.write(data_json.encode('UTF-8'))

if __name__ == '__main__':
    HOST = '192.168.0.209'
    PORT = 8888
    server = socketserver.TCPServer((HOST, PORT), TrainServer)

    server.serve_forever()
