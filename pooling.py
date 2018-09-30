import numpy as np

# Camada de pooling
class pooling:
    def __init__(self, inputs, filter_size, stride):
        self.F = filter_size
        self.S = stride
        self.inputs = inputs
        self.num_inputs = int(inputs.shape[0])
        self.output_size = int((inputs.shape[1] - filter_size)/stride + 1)
    
    # Realiza pooling em uma única imagem
    def pooling_single(self, img):
        output = np.zeros((self.output_size, self.output_size))
        
        # Stride vertical
        for i in range(self.output_size):
            # Stride horizontal
            for j in range(self.output_size):
                output[i, j] = np.amax(img[(i*self.S):(self.F + i*self.S), (j*self.S):(self.F + j*self.S)])
        return output
    
    # Realiza max pooling no conjunto de imagens definido no construtor
    def pooling_max(self):
        subsample = np.zeros((self.num_inputs, self.output_size, self.output_size))
        for i in range(self.num_inputs):
            subsample[i] = pooling_single(self.inputs[i])
        return subsample