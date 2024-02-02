# coding: utf-8
import sys, os
from typing import Callable, Optional
from numbers import Number
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from common.functions import *
from common.gradient import numerical_gradient
import numpy as np
from numpy.typing import NDArray


class NeuralNetwork:
    def __init__(self, layer_count: int, layer_sizes: NDArray, activation_funcs: list[Callable[[NDArray], NDArray]] = None, weight_init_std=0.01):
        self.__activation_funcs = activation_funcs
        if not self.__activation_funcs:
            self.__activation_funcs = [sigmoid if i == layer_count-1 else softmax  for i in range(layer_count)]
        
        # 重みの初期化
        self.params: list[NDArray] = []
        # 入力層があるのでlayerよりひとつ層の数が多くなる
        if layer_count + 1 != len(layer_sizes):
            raise ValueError("layer_sizesはlayer_count+1のサイズである必要があります。")
        
        for i in range(layer_count):
            layer = weight_init_std * np.random.randn(layer_sizes[i], layer_sizes[i+1])
            # 最後の行にバイアスを追加
            layer = np.vstack([layer, np.zeros(layer_sizes[i+1])])
            self.params.append(layer)
            
    def predict(self, inputs: NDArray, weights_matrix:list[ NDArray]):
        """
        inputs: 入力値
                複数行可
                [
                    [1, 12, 31, 0], // 1つ目の入力
                    [78, 13, 134, 7], // 2つ目の入力
                ]
                => 
                [
                    [0.11, 0.44, 0.19, 0.0], // 1つ目の結果
                    [0.19, 0.021, 0.5, 0.0], // 1つ目の結果
                ]
                
        """
        x =  inputs
        for i, weights in enumerate(weights_matrix):
            x = np.hstack([x, np.full((x.shape[0], 1), 1)]) # 最後の列にバイアスを追加
            a = np.dot(x, weights)
            y = self.__activation_funcs[i](a)
            x = y
        return y
        
    # x:入力データ, t:教師データ
    def loss(self, x, t, weights_matrix: Optional[list[NDArray]]=None):
        weights_matrix = weights_matrix or self.params
        y = self.predict(x, weights_matrix)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x, self.params)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # # x:入力データ, t:教師データ
    # def numerical_gradient(self, x, t):
    #     loss_W = lambda W: self.loss(x, t)
        
    #     grads = {}
    #     grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
    #     grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
    #     grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
    #     grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
    #     return grads
    
    def update_weights(self, gradients: list[NDArray], learning_rate = 0.1):
        for i, layer_weights in enumerate(self.params):
            self.params[i] = layer_weights - (learning_rate * gradients[i])

    def gradient(self, x, t):
        """
        x: 入力の行列
            [
                [1, 12, 31, 0], // 1つ目の入力
                [78, 13, 134, 7], // 2つ目の入力
            ]
        
        """
        loss_W = lambda weights_matrix: self.loss(x, t, weights_matrix)
        grads: list[NDArray] = []
        for layer, weights in enumerate(self.params):
            grads.append(self.partial_difference(loss_W, layer, self.params))
        return grads
    
    def _center_difference(self, func: Callable[[Number], Number], x: Number, h: Number = 1e-4) -> Number:
        diff_plus = func(x + h)
        diff_minus = func(x - h)
        return (diff_plus - diff_minus) / (2 * h)
    
    def partial_difference(self, func: Callable[[NDArray], Number], layer: int, weights_matrix: list[NDArray], h: Number = 1e-4) -> Number:
        grad = np.zeros_like(weights_matrix[layer])
        iterator = np.nditer(weights_matrix[layer], flags=['multi_index'], op_flags=['readwrite'])
        while not iterator.finished:
            idx = iterator.multi_index
            def wrap(z) -> Number:
                weights_matrix_copy = [ weights.copy() for weights in weights_matrix]
                weights_matrix_copy[layer][idx] = z
                return func(weights_matrix_copy)
            grad[idx] = self._center_difference(wrap, weights_matrix[layer][idx], h)
            iterator.iternext()
        return grad
    
    # def gradient(self, x, t):
        
        
    #     W1, W2 = self.params['W1'], self.params['W2']
    #     b1, b2 = self.params['b1'], self.params['b2']
    #     grads = {}
        
    #     batch_num = x.shape[0]
        
    #     # forward
    #     a1 = np.dot(x, W1) + b1
    #     z1 = sigmoid(a1)
    #     a2 = np.dot(z1, W2) + b2
    #     y = softmax(a2)
        
    #     # backward
    #     dy = (y - t) / batch_num
    #     grads['W2'] = np.dot(z1.T, dy)
    #     grads['b2'] = np.sum(dy, axis=0)
        
    #     dz1 = np.dot(dy, W2.T)
    #     da1 = sigmoid_grad(a1) * dz1
    #     grads['W1'] = np.dot(x.T, da1)
    #     grads['b1'] = np.sum(da1, axis=0)

    #     return grads
# NeuralNetwork(3, np.array([3, 4, 3, 2])).predict(np.array([2,3,1]))

