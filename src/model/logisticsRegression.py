import numpy as np


class logiticsRgression():

        def __init__(self):
            pass



        def config(self):
            pass

        def sigmoid(self,z):
            return 1/(1+np.exp(z))



        def sign(self,x,shape):
            theta = np.ones(shape)
            return x*theta



        def lossfunction(self,y_t,y_p,name = 'cross_entropy'):
            """
            极大似然估计  对应的梯度上升， 交叉熵对应梯度下降，一个负号的差别
            :param y_t:
            :param y_p:
            :param name:
            :return:
            """
            L =  0
            if name == 'cross_entropy':
                L = - (y_t*np.log(self.sigmoid(y_p)) - (1-y_t)*np.log(1-self.sigmoid(y_p)))

            return L
        def gradient(self,x,h,y):
            return np.dot(x.T, (h - y)) / y.shape[0]

        def fit(self,x,y,lr=0.05, steps = 200,optimatizer='grad_descent'):
            #intercept = np.ones((x.shape[0], 1))  # 初始化截距为 1
            #x = np.concatenate((intercept, x), axis=1)
            w = np.zeros(x.shape[1])  # 初始化参数为 0
            l = float("inf")

            if optimatizer == 'grad_descent':
                for i in range(steps):  # 梯度下降迭代
                    z = np.dot(x, w)  # 线性函数
                    h = self.sigmoid(z)
                    g = self.gradient(x, h, y)  # 计算梯度
                    w -= lr * g  # 通过学习率 lr 计算步长并执行梯度下降
                    l = self.lossfunction(h, y)  # 计算损失函数值
                    if i%100 == 0:
                        print(l)
                    if l < 0.00001:
                        break
            return l,w



