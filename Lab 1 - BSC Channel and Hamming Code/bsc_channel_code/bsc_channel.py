import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import random

class HammingEncoder:
    def __init__(self, n, k):
        self.n = n # length of codeword
        self.k = k # length of message

        self.G = self.generateG() # generator matrix
        # print("G matrix:")
        # print(self.G.shape)
        # print(self.G)

        self.Ht = self.generateHt() # transpose of H
        # print("Ht matrix:")
        # print(self.Ht.shape)
        # print(self.Ht)

    # Here, we will implement the first element as a sum of 1s and the elements in the power of 2s as counters mod 2 for specific regions of the matrix
    # generate generator matrix
    def generateG(self):
        G = np.zeros((self.k, self.n), dtype=int)

        for i in range(self.k):
            for j in range(self.n):
                if i == j:
                    G[i][j] = 1
                elif j >= self.k:
                    G[i][j] = 1
                    if (i == 3 and j == 4 or 
                        i  == 2 and j == 6 or 
                        i == 1 and j == 5): G[i][j] = 0
        return G

    # generate parity-check matrix
    def generateHt(self):
        Ht = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=int)
        return Ht

    def encode(self, message):
        # encode message
        return message @ self.G % 2

    def decode(self, received):
        # decode received message
        syndrome = received @ self.Ht % 2
        # print("Syndrome:")
        # print(syndrome)

        syndrome_dec = syndrome[0, 0] * 2**2 + syndrome[0, 1] * 2 ** 1 + syndrome[0, 2] * 2 ** 0    # Convert syndrome to decimal
        e = np.zeros((1, self.n), dtype=int)

        # Possible errors based on the syndrome value (defining the minimum number of errors == 1)
        if   (syndrome_dec == 1): e[0, 6] = 1   # 7a position error
        elif (syndrome_dec == 2): e[0, 5] = 1   # 6a position error
        elif (syndrome_dec == 3): e[0, 3] = 1   # 4a position error
        elif (syndrome_dec == 4): e[0, 4] = 1   # 5a position error 
        elif (syndrome_dec == 5): e[0, 1] = 1   # 2nd position error
        elif (syndrome_dec == 6): e[0, 2] = 1   # 3rd position error
        elif (syndrome_dec == 7): e[0, 0] = 1   # 1st position error
    
        decoded = (e + received) % 2
        return decoded
    
    def correctError(self, received):
        # Correct any single-bit error in the received codeword
        pass

class MyEncoder:
    def __init__(self, n, k):
        self.n = n # length of codeword
        self.k = k # length of message

        self.G = self.generateG() # generator matrix
        # print("G matrix:")
        # print(self.G.shape)
        # print(self.G)

        self.Ht = self.generateHt() # transpose of H
        # print("Ht matrix:")
        # print(self.Ht.shape)
        # print(self.Ht)

    # Here, we will implement the first element as a sum of 1s and the elements in the power of 2s as counters mod 2 for specific regions of the matrix
    # generate generator matrix
    def generateG(self):
        G = np.array([
        #   b1 b2 b3 b4 b5 p1 p2 p3 p4
            [1, 0, 0, 0, 0, 1, 1, 1, 1],    # b1
            [0, 1, 0, 0, 0, 1, 1, 1, 0],    # b2
            [0, 0, 1, 0, 0, 1, 1, 0, 1],    # b3
            [0, 0, 0, 1, 0, 1, 0, 1, 1],    # b4
            [0, 0, 0, 0, 1, 0, 1, 1, 1]     # b5

        ], dtype=int)
        return G

    # generate parity-check matrix
    def generateHt(self):
        Ht = np.array([
            [1, 1, 1, 1],   # b1
            [1, 1, 1, 0],   # b2
            [1, 1, 0, 1],   # b3
            [1, 0, 1, 1],   # b4
            [0, 1, 1, 1],   # b5
            [1, 0, 0, 0],   # p1
            [0, 1, 0, 0],   # p2
            [0, 0, 1, 0],   # p3
            [0, 0, 0, 1]    # p4
        ], dtype=int)
        return Ht

    def encode(self, message):
        # encode message
        return message @ self.G % 2

    def decode(self, received):
        # decode received message
        syndrome = received @ self.Ht % 2
        # print("Syndrome:")
        # print(syndrome)

        syndrome_dec = syndrome[0, 0] * 2**3 + syndrome[0, 1] * 2 ** 2 + syndrome[0, 2] * 2 ** 1 + syndrome[0, 3] * 2 ** 0    # Convert syndrome to decimal
        e = np.zeros((1, self.n), dtype=int)

        # Possible errors based on the syndrome value (defining the minimum number of errors == 1)
        if   (syndrome_dec == 15): e[0, 0] = 1  # 1st position error
        elif (syndrome_dec == 14): e[0, 1] = 1  # 2nd position error
        elif (syndrome_dec == 13): e[0, 2] = 1  # 3rd position error
        elif (syndrome_dec == 11): e[0, 3] = 1  # 4th position error
        elif (syndrome_dec ==  7): e[0, 4] = 1  # 5th position error
        elif (syndrome_dec ==  8): e[0, 5] = 1  # 6th position error
        elif (syndrome_dec ==  4): e[0, 6] = 1  # 7th position error
        elif (syndrome_dec ==  2): e[0, 7] = 1  # 8th position error
        elif (syndrome_dec ==  1): e[0, 8] = 1  # 9th position error
    
        decoded = (e + received) % 2
        return decoded
    
    def correctError(self, received):
        # Correct any single-bit error in the received codeword
        pass


class BSC:
    def __init__(self, p):
        self.p = p # probability of bit flip

    def transmit(self, codewords):
        # received = codeword.copy()

        # transmit codeword.
        flip_mask = np.random.rand(*codewords.shape) < self.p
        receiveds = np.mod(codewords + flip_mask, 2)

        # # transmit codeword.
        # for i in range(0, received.shape[1]):
        #     if (random.random() < self.p):
        #         received[0, i] = (received[0, i] + 1) % 2

        print(receiveds.shape)
        print(receiveds[:, :60])
        return receiveds

# Generate a random x-bits message
def generateInformationBits(x):
    return np.random.randint(0, 2, x).astype(int).reshape(1,x)

# Example using the Hamming(7, 4) code 1 time
def example():
    # Example parameters for Hamming(15, 11) code
    n = 7 # length of codeword
    k = 4 # length of message
    p = 0.15 # probability of bit flip

    hamming = HammingEncoder(n, k)
    channel = BSC(p)

    # Example usage
    message = generateInformationBits(k)
    print("Original message:", message)

    codeword = hamming.encode(message)
    print("Encoded message: ", codeword)

    received = channel.transmit(codeword)
    print("Received message:", received)

    decoded = hamming.decode(received)
    print("Decoded message:", decoded)

def plot_error_rates_with_spline(p, error_rates0, error_rates1, error_rates2, y_target):
    plt.figure(figsize=(10, 6))

    # Dados para plotagem
    datasets = [error_rates0, error_rates1, error_rates2]
    colors = ['b', 'r', 'g']
    labels = ['No Treatment', 'Hamming(7, 4) Code', 'My Encoder(9, 5) Code']

    # Interpolar e plotar cada conjunto de dados
    for data, color, label in zip(datasets, colors, labels):
        # Criar um objeto Spline com um fator de suavização
        spline = UnivariateSpline(p, data, s=len(p))  # s é um parâmetro de suavização

        # Gerar pontos x para a curva suave
        x_smooth = np.logspace(np.log10(min(p)), np.log10(max(p)), 500)
        y_smooth = spline(x_smooth)

        # Plotar a curva suave
        plt.plot(x_smooth, y_smooth, color=color, linestyle='-', marker='o', markersize=2, label=label)

        # Encontrar x onde y é aproximadamente igual a y_target
        # roots = spline.roots() - y_target  # Encontrar raízes onde y - y_target = 0
        # for root in roots:
        #     if np.isreal(root) and min(p) <= root <= max(p):
        #         plt.axvline(x=root.real, color=color, linestyle='--', alpha=0.7)
        #         plt.text(root.real, y_target * 1.1, f'{root.real:.4f}', ha='center')

    plt.xlabel('Probability of bit flip (p)')
    plt.ylabel('Error Rate')
    plt.title('Error Rate vs. Channel Noise for Different Coding Schemes')
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(p, labels=[f'{pp:.5f}' for pp in p])
    plt.xlim(max(p), min(p))  # Invert the x-axis to show the values in ascending order
    plt.yticks([0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005], labels=['0.5', '0.2', '0.1', '0.05', '0.02', '0.01', '0.005', '0.002', '0.001', '0.0005', '0.0002', '0.0001', '0.00005'])
    plt.ylim(0.00005, 0.5)
    plt.legend(title="Legend", title_fontsize='13', fontsize='11')
    plt.grid(True)
    plt.show()

# Simulate and Compare methods for different values of p
def simulate(x_info_bits, p):
    # No treatment case
    k0 = 100
    n0 = 100

    # Parameters for Hamming(7, 4) code
    n1 = 7 # length of codeword
    k1 = 4 # length of message
    encoder1 = HammingEncoder(n1, k1)

    # Parameters for my encoder (9, 5) code
    n2 = 9 # length of codeword
    k2 = 5 # length of message
    encoder2 = MyEncoder(n2, k2)

    message = generateInformationBits(x_info_bits)
    print("Original message:", message[:, :60])
    print(message.shape)

    # Case 0 - No treatment case
    encoded_blocks0 = message.reshape(x_info_bits // n0, 1, n0)

    # Case 1 - Divide message into blocks of 4 bits and encode each one separately for blocks of 7 bits
    encoded_blocks1 = encoder1.encode([message[:, i:i+k1] for i in range(0, message.shape[1], k1)])
    print("Shape of encoded blocks:", encoded_blocks1.shape)
    print("Encoded message:", encoded_blocks1[:, :60])

    # Case 2 - Divide message into blocks of 5 bits and encode each one separately for blocks of 9 bits
    encoded_blocks2 = encoder2.encode([message[:, i:i+k2] for i in range(0, message.shape[1], k2)])
    print("Shape of encoded blocks:", encoded_blocks2.shape)
    print("Encoded message:", encoded_blocks2[:, :60])

    # Declaring the necessary variables to store the final message and error rates
    error_rates0 = []
    error_rates1 = []
    error_rates2 = []
    
    # Transmit each block through the channel for different values of p
    print("\nTRANSMISSION THROUGH BSC\n")
    for i in range(0, len(p)):
        print("\n\tp = ", p[i], "\n")
        channel = BSC(p[i])

        # Initialize the necessary np arrays to store the final message
        final_message0 = np.empty((1, 0), dtype=int)
        final_message1 = np.empty((1, 0), dtype=int)
        final_message2 = np.empty((1, 0), dtype=int)

        # Case 0 - Transmit the endoded blocks 0 through the channel (all at once)
        received0 = channel.transmit(encoded_blocks0)

        # Case 1 - Transmit the endoded blocks 1 through the channel (all at once)
        received1 = channel.transmit(encoded_blocks1)
        print("Shape of received message:", received1.shape)
        print("Received message:", received1[:, :60])

        # Case 2 - Transmit the endoded blocks 2 through the channel (all at once)
        received2 = channel.transmit(encoded_blocks2)
        print("Shape of received message:", received2.shape)
        print("Received message:", received2[:, :60])

        # Case 0 - no decode needed
        for j in range(0, len(encoded_blocks0)):
            relevant_bits0 = received0[j, :, :k0]
            final_message0 = np.concatenate((final_message0, relevant_bits0), axis=1)

        # Case 1 - Decode the received blocks
        for j in range(0, len(encoded_blocks1)):
            decoded1 = encoder1.decode(received1[j, :, :])
            relevant_bits1 = decoded1[:, :k1]    # Extract only the first four information bits of decoded1
            final_message1 = np.concatenate((final_message1, relevant_bits1), axis=1)

        # Case 2 - Decode the received blocks
        for j in range(0, len(encoded_blocks2)):
            decoded2 = encoder2.decode(received2[j, :, :])
            relevant_bits2 = decoded2[:, :k2]
            final_message2 = np.concatenate((final_message2, relevant_bits2), axis=1)

        print("Final message0:", final_message0[:, :60])
        print("Final message1:", final_message1[:, :60])
        print("Final message2:", final_message2[:, :60])

        # Calculate the number of different bits for each case
        # Case 0
        bit_errors0 = np.sum(message != final_message0)
        error_rate0 = bit_errors0 / x_info_bits
        error_rates0.append(error_rate0)
        print("Error rate0: ", error_rate0)

        # Case 1
        bit_errors1 = np.sum(message != final_message1)
        error_rate1 = bit_errors1 / x_info_bits
        error_rates1.append(error_rate1)
        print("Error rate1: ", error_rate1)

        # Case 2
        bit_errors2 = np.sum(message != final_message2)
        error_rate2 = bit_errors2 / x_info_bits
        error_rates2.append(error_rate2)
        print("Error rate2: ", error_rate2)

    # Plot the error rates
    plt.figure(figsize=(10, 6))
    plt.plot(p, error_rates0, 'b-o', label='No Treatment')
    plt.plot(p, error_rates1, 'r-o', label='Hamming(7, 4) Code')
    plt.plot(p, error_rates2, 'g-o', label='My Encoder(9, 5) Code')
    plt.scatter(p, error_rates1, color='red')  # mark each point
    plt.xlabel('Probability of bit flip (p)')
    plt.ylabel('Error Rate')
    plt.title('Error Rate vs. Channel Noise for Different Coding Schemes')
    plt.xscale('log')
    plt.xticks(p, labels=[str(pp) for pp in p])
    plt.xlim(max(p), min(p))  # Invert the x-axis to show the values in ascending order
    plt.yscale('log')
    plt.yticks([0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005], labels=['0.5', '0.2', '0.1', '0.05', '0.02', '0.01', '0.005', '0.002', '0.001', '0.0005', '0.0002', '0.0001', '0.00005'])
    plt.ylim(0.00005, 0.5)
    plt.legend(title="Legend", title_fontsize='13', fontsize='11')  # Add a title to the legend for clarity
    plt.grid(True)
    plt.show()

    # plot_error_rates_with_spline(p, error_rates0, error_rates1, error_rates2, 0.0001)

# Objective: Simulate a binary symmetric channel (BSC) with a Hamming code
def main():
    # example()
    simulate(x_info_bits=1000000, p=[0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])


main()