import random
import math

class SimpleRNN:
    def __init__(self, input_size, hidden_size):
        self.w_xh = [[random.uniform(-0.5,0.5) for _ in range(input_size)] for _ in range(hidden_size)]
        self.w_hh = [[random.uniform(-0.5,0.5) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.w_hy = [random.uniform(-0.5,0.5) for _ in range(hidden_size)]
        self.b_h = [0.0]*hidden_size
        self.b_y = 0.0

    def _sigmoid(self, z):
        return 1/(1+math.exp(-z))

    def _forward(self, seq):
        h = [0.0]*len(self.w_xh)
        for x in seq:
            new_h = []
            for j in range(len(self.w_xh)):
                z = sum(w*x_i for w,x_i in zip(self.w_xh[j],[x])) + sum(w*h_i for w,h_i in zip(self.w_hh[j],h)) + self.b_h[j]
                new_h.append(math.tanh(z))
            h = new_h
        y = self._sigmoid(sum(w*h_i for w,h_i in zip(self.w_hy,h)) + self.b_y)
        return h, y

    def predict(self, seq):
        _, y = self._forward(seq)
        return 1 if y >= 0.5 else 0

    def train(self, data, labels, epochs=500, lr=0.1):
        for _ in range(epochs):
            for seq, label in zip(data, labels):
                h_states = [[0.0]*len(self.w_xh)]
                h = [0.0]*len(self.w_xh)
                for x in seq:
                    new_h = []
                    for j in range(len(self.w_xh)):
                        z = sum(w*x_i for w,x_i in zip(self.w_xh[j],[x])) + sum(w*h_i for w,h_i in zip(self.w_hh[j],h)) + self.b_h[j]
                        new_h.append(math.tanh(z))
                    h = new_h
                    h_states.append(h)
                y = self._sigmoid(sum(w*h_i for w,h_i in zip(self.w_hy,h)) + self.b_y)
                dy = y - label
                # output gradients
                for j in range(len(self.w_hy)):
                    self.w_hy[j] -= lr * dy * h[j]
                self.b_y -= lr * dy
                # hidden gradients (backprop through time up to seq len)
                dh_next = [0.0]*len(self.w_xh)
                for t in reversed(range(len(seq))):
                    dh = [dh_next[j] + dy*self.w_hy[j]*(1-h_states[t+1][j]**2) for j in range(len(self.w_xh))]
                    for j in range(len(self.w_xh)):
                        self.b_h[j] -= lr * dh[j]
                        self.w_xh[j][0] -= lr * dh[j]*seq[t]
                        for k in range(len(self.w_xh)):
                            self.w_hh[j][k] -= lr * dh[j]*h_states[t][k]
                    dh_next = [sum(self.w_hh[k][j]*dh[k] for k in range(len(self.w_xh))) for j in range(len(self.w_xh))]

if __name__ == "__main__":
    data = [[0,1,0],[1,0,1],[1,1,1],[0,0,0]]
    labels = [1,1,0,0]  # parity of ones mod 2
    rnn = SimpleRNN(input_size=1, hidden_size=2)
    rnn.train(data, labels, epochs=200)
    for seq in data:
        pred = rnn.predict(seq)
        print(f"Seq: {seq} => Predicted: {pred}")
