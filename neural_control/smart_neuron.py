import numpy as np

from neuron import NeuralNetwork


class NeuralNetworkNumPy(NeuralNetwork):
    """ニューラルネットワークの学習管理クラス"""
    LEARNING_RATE = 1e-3  # 学習率

    def __init__(self, network_file_path):
        super().__init__(network_file_path)
        # list -> ndarray
        self.params['W_INPUT'] = np.array(self.params['W_INPUT'])
        self.params['W_HIDDEN'] = np.array(self.params['W_HIDDEN'])
        self.params_delta = {
            'W_INPUT': np.zeros((self.INPUT_LAYER_NEURONS, self.HIDDEN_LAYER_NEURONS)),
            'W_HIDDEN': np.zeros((self.HIDDEN_LAYER_NEURONS, self.OUTPUT_LAYER_NEURONS)),
        }

    def forward(self, x_input, should_save_output=False):
        """ネットワークを順伝搬させて出力を計算する

        入力層のニューロンは順に1, 2, ..., i, ...
        隠れ層と出力層も同様にj, kと添字をふることにする
        ここではuは入力値と重みの総和、φは任意の活性化関数、yはニューロンの出力とする
        φ_hは隠れ層の活性化関数。ここではReLUを使う
        φ_oは隠れ層の活性化関数。ここでは恒等関数を使う※1

        numpyのndarrayを使って行列演算する
        コメントの数式は1変数ずつの計算を書いているが、コードは層ごとに一括の計算であることに注意
        """
        # 隠れ層の計算
        # u_j = Σ_i { x_i * w_ij }
        u_hidden = np.dot(x_input, self.params['W_INPUT'])
        # y_j = φ_h(u_j)
        y_hidden = self.relu(u_hidden)  # 活性化関数はReLU

        # 出力層の計算
        # u_k = Σ_j { y_j * w_jk }
        u_output = np.dot(y_hidden, self.params['W_HIDDEN'])
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
        delta_o = y_output - target
        delta_w2 = np.outer(y_hidden, delta_o)
        self.params_delta['W_HIDDEN'] += -self.LEARNING_RATE * delta_w2

        # 入力層 - 隠れ層間の重みを更新
        # φ_h'(u_j)はReLUの微分
        delta_relu = u_hidden > 0
        # delta_w1_tmpは　Σ_k{ δ_output_k * w_jk } * φ_h'(u_j) までの計算
        delta_w1_tmp = np.dot(self.params['W_HIDDEN'], delta_o) * delta_relu
        delta_w1 = np.outer(x_input, delta_w1_tmp)
        self.params_delta['W_INPUT'] += -self.LEARNING_RATE * delta_w1

    @staticmethod
    def relu(inputs):
        """活性化関数ReLU"""
        outputs = np.array(inputs)
        outputs[outputs < 0] = 0
        return outputs
