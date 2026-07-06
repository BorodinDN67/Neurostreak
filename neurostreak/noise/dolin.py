import numpy as np
import scipy as sc
#TODO Чота модель сполпинка найти и построить не удалось, аппелирую к более зрелой версии себя заняться этим

#Класс будет содержать все необходимые для модели долина параметры
#Очевидно, сделано это с целью наличия экспериментов и возможной автоматизации подбора параметров

#Просто контейнер для параметров симуляции
class Water:
    def __init__(self, absorption, scattering,indicatrix):
        self.absorption = absorption
        self.scattering = scattering
        self.indicatrix = indicatrix

        self.velocity = 2.23826 * 10 ** 8 #м/с

class MTF_Dolin:
    def __init__(self, water, k, gamma, theta0, k_theta, L, mu):
        self.water = water
        self.absorption = water.absorption
        self.scattering = water.scattering
        #self.indicatrix = water.indicatrix
        self. velocity = water.velocity


        self.k_theta = k_theta
        self.L = L
        self.mu = mu
        self.theta0 = theta0
        self.gamma = gamma
        self.k = k

    def kernal_dolin(self, distance):

        n = np.arange(self.L, dtype = np.float64)

        rho = 1 - np.exp(-self.gamma * distance * self.scattering)
        theta = self.theta0 + self.k_theta * distance * self.scattering * self.mu
        g = (n ** (self.k - 1)) * np.exp(-n / theta)
        g = g / g.sum()
        delta = np.zeros(self.L, dtype=np.float64)
        delta[0] = 1.0

        q = (1.0 - rho) * delta + rho * g
        q = q / q.sum()

        return q

    def simulate(self, template, distance):
        q = self.kernal_dolin(distance)
        amp = np.exp(-2*(self.absorption + self.scattering) * distance)

        return amp * np.convolve(template, q, mode='same')