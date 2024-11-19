import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class PenduloSimple:
    def __init__(self, m=1.0, l=1.0, g=9.81, b=0.1):
        """
        Inicializa el péndulo simple
        m: masa (kg)
        l: longitud (m)
        g: aceleración gravitacional (m/s^2)
        b: coeficiente de fricción
        """
        self.m = m  # masa
        self.l = l  # longitud
        self.g = g  # gravedad
        self.b = b  # fricción
        
    def ecuaciones_estado(self, estado, t, u=0):
        """
        Ecuaciones de estado del péndulo
        estado[0] = theta (ángulo)
        estado[1] = theta_dot (velocidad angular)
        u = torque de control
        """
        theta = estado[0]
        theta_dot = estado[1]
        # Ecuaciones diferenciales
        dtheta = theta_dot
        dtheta_dot = (-self.b*theta_dot - self.m*self.g*self.l*np.sin(theta) + u)/(self.m*self.l**2)
        
        return [dtheta, dtheta_dot]
    
    def simular(self, estado_inicial, t_span, u=0):
        """
        Simula el sistema
        estado_inicial: [theta_0, theta_dot_0]
        t_span: vector de tiempo
        u: entrada de control (puede ser escalar o función)
        """
        if callable(u):
            sol = odeint(lambda x, t: self.ecuaciones_estado(x, t, u(x, t)), estado_inicial, t_span)
        else:
            sol = odeint(lambda x, t: self.ecuaciones_estado(x, t, u), estado_inicial, t_span)
        return sol
    
    def parametros(self):
        """
        Devuelve los parámetros del sistema
        """
        return {'m': self.m, 'l': self.l, 'g': self.g, 'b': self.b}