import numpy as np
import mnist
import time

# Baixa o banco de dados. Rode apenas se não possuir o arquivo "mnist.pkl"
# mnist.init()

# Carregando os dados
x_train, t_train, x_test, t_test = mnist.load()

# Transformando dataset em um vetor de imagens 28x28
X = np.reshape(x_train, (60000,28, 28))

""" 
 *	Camada de convolução:
 *		Possui duas funções que realizam a convolução de imagens com filtros
 * 		(funções "conv_full" e "conv_dot"), sendo que a última possui melhor performance.
"""
class conv_layer:
    def __init__(self, inputs, filters, padding, stride):
        self.inputs = inputs
        self.filters = filters
        self.num_inputs = int(inputs.shape[0])
        self.num_filters = int(filters.shape[0])
        self.filter_size = int(filters.shape[1])
        self.S = stride
        self.P = padding
        self.output_size = int((inputs.shape[1] - self.filter_size + 2*padding)/stride + 1)
    
    
    # Função que convoluciona uma imagem para um determinado filtro. Retorna uma matriz de convolução
    def conv_single(self, img, filtro):
        if self.P > 0:
            img_pad = np.pad(array=img, pad_width=self.P, mode='constant', constant_values=0)
        else:
            img_pad = img
        output = np.zeros((self.output_size, self.output_size))
        
        # Stride vertical
        for i in range(self.output_size):
            # Stride horizontal
            for j in range(self.output_size):
                # Geração da submatriz através de slicing
                img_sub = img_pad[(i*self.S):(self.filter_size + i*self.S), (j*self.S):(self.filter_size + j*self.S)]
                output[i, j] = np.sum(img_sub*filtro)
        return output
    
    # Convoluciona conjunto de imagens (inputs) com conjunto de filtros. Retorna conjunto de matrizes de convolução
    def conv_full(self):
        outputs = np.zeros((self.num_inputs*self.num_filters, self.output_size, self.output_size))
        k = 0
        for i in range(self.num_inputs):
            for j in range(self.num_filters):
                outputs[k] = self.conv_single(self.inputs[i], self.filters[j])
                k += 1     
        return outputs
        
    
    # Transforma imagem em uma matriz em que cada coluna é um campo receptivo alongado
    def img2col(self, img): # (28, 28)
        if self.P > 0:
            img_pad = np.pad(array=img, pad_width=self.P, mode='constant', constant_values=0)
        else:
            img_pad = img
        img_col = np.zeros((self.filter_size**2, self.output_size**2))
        k = 0
        
        # Stride vertical
        for i in range(self.output_size):
            # Stride horizontal
            for j in range(self.output_size):
                # Geração da submatriz através de slicing
                img_col[:, k] = np.ravel(img_pad[(i*self.S):(self.filter_size + i*self.S), (j*self.S):(self.filter_size + j*self.S)])
                k += 1        
        return img_col
    
    # Método de multiplicação matricial, usando img2col
    def conv_dot(self):
        W_row = np.reshape(self.filters, (self.num_filters, self.filter_size**2)) # (3, 9)
        output_dot = np.zeros((self.num_inputs, self.filter_size, self.output_size**2)) # (60000, 3, 100)
        
        for i in range(self.num_inputs): # 60000
            output_dot[i] = np.dot(W_row, self.img2col(self.inputs[i])) # (3, 9) x (9, 100) = (3, 100)
        
        # Retornando resultado com suas dimensoões originais
        return np.reshape(output_dot, (self.num_inputs*self.num_filters, self.output_size, self.output_size)) # (18000, 10, 10)
