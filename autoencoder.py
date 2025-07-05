import random
import math

class Autoencoder:
    def __init__(self, n_inputs, n_hidden):
        self.w1 = [[random.uniform(-0.5,0.5) for _ in range(n_inputs)] for _ in range(n_hidden)]
        self.b1 = [0.0]*n_hidden
        self.w2 = [[random.uniform(-0.5,0.5) for _ in range(n_hidden)] for _ in range(n_inputs)]
        self.b2 = [0.0]*n_inputs

    def _sigmoid(self, z):
        return 1/(1+math.exp(-z))

    def _sigmoid_deriv(self, a):
        return a*(1-a)

    def encode(self, x):
        hidden = []
        for w,b in zip(self.w1,self.b1):
            z = sum(w_i*x_i for w_i,x_i in zip(w,x)) + b
            hidden.append(self._sigmoid(z))
        return hidden

    def decode(self, h):
        out = []
        for w,b in zip(self.w2,self.b2):
            z = sum(w_i*h_i for w_i,h_i in zip(w,h)) + b
            out.append(self._sigmoid(z))
        return out

    def train(self, X, epochs=1000, lr=0.1):
        for _ in range(epochs):
            for x in X:
                h = self.encode(x)
                out = self.decode(h)
                # output error
                delta_out = [o - x_i for o,x_i in zip(out,x)]
                # hidden error
                delta_h = []
                for j in range(len(self.w1)):
                    err = sum(self.w2[i][j]*delta_out[i] for i in range(len(x)))
                    delta_h.append(err*self._sigmoid_deriv(h[j]))
                # update output weights
                for i in range(len(self.w2)):
                    for j in range(len(self.w2[i])):
                        self.w2[i][j] -= lr * delta_out[i] * h[j]
                    self.b2[i] -= lr * delta_out[i]
                # update input weights
                for j in range(len(self.w1)):
                    for i in range(len(self.w1[j])):
                        self.w1[j][i] -= lr * delta_h[j]*x[i]
                    self.b1[j] -= lr * delta_h[j]

if __name__ == "__main__":
    X = [[0,0],[0,1],[1,0],[1,1]]
    ae = Autoencoder(n_inputs=2, n_hidden=2)
    ae.train(X, epochs=500)
    for x in X:
        h = ae.encode(x)
        out = ae.decode(h)
        print(f"Input: {x} => Reconstructed: {[round(o,2) for o in out]}")
