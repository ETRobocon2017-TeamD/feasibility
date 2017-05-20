import pickle


class NeuralNetwork(object):
    """ニューラルネットワークの学習管理クラス"""
    INPUT_LAYER_NEURONS = 4  # 入力層ニューロン数
    HIDDEN_LAYER_NEURONS = 16  # 隠れ層ニューロン数
    OUTPUT_LAYER_NEURONS = 5  # 出力層ニューロン数 = Action数

    def __init__(self, network_file_path):
        if network_file_path:
            with open(network_file_path, 'rb') as file:
                self.params = pickle.load(file)
        else:
            self.params = {}
        self.output = None

    def update_network(self, params):
        self.params = params

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
