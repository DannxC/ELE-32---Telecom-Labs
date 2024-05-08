import numpy as np
import matplotlib.pyplot as plt
import random   # For random number generation

class Graph:
    def __init__(self):
        self.adj_list = {}   # {vertex: [neighbours]}

    def add_vertex(self, vertex):
        if vertex not in self.adj_list:
            self.adj_list[vertex] = []

    def add_edge(self, vertex1, vertex2):
        if vertex1 in self.adj_list and vertex2 in self.adj_list:
            self.adj_list[vertex1].append(vertex2)
            self.adj_list[vertex2].append(vertex1)  # Comment this line for a directed graph

    def display(self):
        for vertex in self.adj_list:
            print(f"{vertex}: {self.adj_list[vertex]}")

class LDPCEncoder:
    # Initialize LDPC Encoder with parameters
    # n: number of variable nodes (or codeword length)
    # dv: degree of each variable node
    # dc: degree of each check node
    def __init__(self, n: int, dv: int, dc: int):
        # Calculate number of check nodes and check if it is integer
        self.m: int = n * dv // dc
        if n * dv % dc != 0:
            print("Invalid parameters: m must be an integer")
            return
        
        # Parameters
        self.n: int = n
        self.dv: int = dv
        self.dc: int = dc

        # Tanner Graph
        self.generate_tanner_graph()

        # G matrix
        self.G = self.generateG() # generator matrix

    def generateG(self):
        G = np.zeros((self.n - self.m, self.n), dtype=int)
        return G

    # Generate Tanner graph
    # From idx 0 to n-1: variable nodes
    # From idx n to n+m-1: check nodes
    def generate_tanner_graph(self):
        redo = True
        while (redo):
            redo = False
            # Create the graph and Add vertices for variable and check nodes
            self.graph = Graph()
            for i in range(self.n + self.m):
                self.graph.add_vertex(i)
                
            # Initialize check nodes and their usage count
            check_nodes = [i + self.n for i in range(self.m)]
            check_nodes_count = {node: 0 for node in check_nodes}

            # Connect variable nodes to check nodes
            for variable_node in range(self.n):
                if not check_nodes:
                    break

                check_nodes_copy = check_nodes.copy()

                for _ in range(self.dv):
                    # Choose a random check node between the ones available on the copy list of check nodes
                    if (len(check_nodes_copy) == 0):
                        redo = True
                        break
                    idx = random.randint(0, len(check_nodes_copy)-1)
                    current_check_node = check_nodes_copy[idx]

                    # Add edge between variable node and check node
                    self.graph.add_edge(variable_node, current_check_node)
                    # Finally pop the idx used for the current_check_node
                    check_nodes_copy.pop(idx)

                    # Update check node count and remove it from the original list if it reaches dc
                    check_nodes_count[current_check_node] += 1
                    if check_nodes_count[current_check_node] == self.dc:
                        check_nodes.remove(current_check_node)
                        check_nodes_count.pop(current_check_node)

    # Encode function       -> FOR NOW IT IS NOT IMPLEMENTED FOR REAL
    def encode(self, message):
        return message @ self.G % 2

    # Decode function
    # received is a vector that represents a group of self.n bits
    def decode(self, received):
        decoded = received.copy()

        max_iterations = 0
        while max_iterations <= 10:
            # Initialize arrays correctly
            parity_checks = np.zeros(self.m, dtype=bool)    # Boolean array of size m
            var_nodes_counter = np.zeros(self.n, dtype=int) # Integer array of size n

            max_value = 0
            
            # Calculate parity checks
            for i in range(self.m):
                # Compute parity checks based on adjacency list in the Tanner graph
                # Here, make sure the indices are within the bounds of the decoded array
                parity_checks[i] = (sum(decoded[0, node] for node in self.graph.adj_list[i + self.n]) % 2) == 0

                if not parity_checks[i]:
                    for node in self.graph.adj_list[i + self.n]:
                        var_nodes_counter[node] += 1
                        if var_nodes_counter[node] > max_value:
                            max_value = var_nodes_counter[node]
            
            # Stop if don't have more bits to flip
            if max_value == 0:
                break

            # Bit flip
            for i in range(self.n):
                if var_nodes_counter[i] == max_value:
                    decoded[0, i] = (decoded[0, i] + 1) % 2

            max_iterations += 1

        return decoded

# BSC (Binary Symmetric Channel) class
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

        return receiveds

# Hamming Encoder class
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

# Generate a 0's x-bits message
def generateInformationBits(x):
    # return np.random.randint(0, 2, x).astype(int).reshape(1,x)
    return np.zeros(x, dtype=int).reshape(1, x)

# Simulate and Compare methods for different values of p    
def simulate(x_info_bits, p):
    # Generate message with x_info_bits
    message = generateInformationBits(x_info_bits)

    # Case 0: HammingCode (7, 4)
    # Parameters
    n0 = 7 # length of codeword
    k0 = 4 # length of message
    hamming_encoder = HammingEncoder(n0, k0)
    encoded_blocks0 = hamming_encoder.encode([message[:, i:i+k0] for i in range(0, message.shape[1], k0)])

    # Case 1: LDPC with n = 98
    n1 = 98
    m1 = n1 * 3 // 7
    ldpc_encoder1 = LDPCEncoder(n1, 3, 7)   # n, dv, dc
    #ldpc_encoder1.graph.display()
    x1_info_bits = message.shape[1]-(message.shape[1] % (n1-m1))
    encoded_blocks1 = ldpc_encoder1.encode([message[:, i:i+n1-m1] for i in range(0, x1_info_bits, n1-m1)])

    # Case 2: LDPC with n = 203
    n2 = 203
    m2 = n2 * 3 // 7
    ldpc_encoder2 = LDPCEncoder(n2, 3, 7)
    # ldpc_encoder2.graph.display()
    x2_info_bits = message.shape[1]-(message.shape[1] % (n2-m2))
    encoded_blocks2 = ldpc_encoder2.encode([message[:, i:i+n2-m2] for i in range(0, x2_info_bits, n2-m2)])

    # Case 3: LDPC with n = 497
    n3 = 497
    m3 = n3 * 3 // 7
    ldpc_encoder3 = LDPCEncoder(n3, 3, 7)
    # ldpc_encoder3.graph.display()
    x3_info_bits = message.shape[1]-(message.shape[1] % (n3-m3))
    encoded_blocks3 = ldpc_encoder3.encode([message[:, i:i+n3-m3] for i in range(0, x3_info_bits, n3-m3)])

    # Case 4: LDPC with n = 1001
    n4 = 1001
    m4 = n4 * 3 // 7
    ldpc_encoder4 = LDPCEncoder(n4, 3, 7)
    # ldpc_encoder4.graph.display()
    x4_info_bits = message.shape[1]-(message.shape[1] % (n4-m4))
    encoded_blocks4 = ldpc_encoder4.encode([message[:, i:i+n4-m4] for i in range(0, x4_info_bits, n4-m4)])

    # Declaring the necessary variables to store the final message and error rates
    error_rates0 = []
    error_rates1 = []
    error_rates2 = []
    error_rates3 = []
    error_rates4 = []
    
    # Transmit each block through the channel for different values of p
    print("\nTRANSMISSION THROUGH BSC\n")
    for i in range(0, len(p)):
        print("\n\tp = ", p[i], "\n")
        channel = BSC(p[i])

        # Initialize the necessary np arrays to store the final message
        final_message0 = np.empty((1, 0), dtype=int)
        final_message1 = np.empty((1, 0), dtype=int)
        final_message2 = np.empty((1, 0), dtype=int)
        final_message3 = np.empty((1, 0), dtype=int)
        final_message4 = np.empty((1, 0), dtype=int)

        # Case 0 - Transmit the encoded blocks 0 through the channel (all at once)
        received0 = channel.transmit(encoded_blocks0)

        # Case 1 - Transmit the encoded blocks 1 through the channel (all at once)
        received1 = channel.transmit(encoded_blocks1)

        # Case 2 - Transmit the encoded blocks 2 through the channel (all at once)
        received2 = channel.transmit(encoded_blocks2)

        # Case 3 - Transmit the encoded blocks 3 through the channel (all at once)
        received3 = channel.transmit(encoded_blocks3)

        # Case 4 - Transmit the encoded blocks 4 through the channel (all at once)
        received4 = channel.transmit(encoded_blocks4)

        # Decode the received blocks for each case
        for j in range(0, len(encoded_blocks0)):
            decoded0 = hamming_encoder.decode(received0[j, :, :])
            relevant_bits0 = decoded0[:, :k0]
            final_message0 = np.concatenate((final_message0, relevant_bits0), axis=1)

        for j in range(0, len(encoded_blocks1)):
            decoded1 = ldpc_encoder1.decode(received1[j, :, :])
            relevant_bits1 = decoded1[:, :n1-m1]
            final_message1 = np.concatenate((final_message1, relevant_bits1), axis=1)

        for j in range(0, len(encoded_blocks2)):
            decoded2 = ldpc_encoder2.decode(received2[j, :, :])
            relevant_bits2 = decoded2[:, :n2-m2]
            final_message2 = np.concatenate((final_message2, relevant_bits2), axis=1)

        for j in range(0, len(encoded_blocks3)):
            decoded3 = ldpc_encoder3.decode(received3[j, :, :])
            relevant_bits3 = decoded3[:, :n3-m3]
            final_message3 = np.concatenate((final_message3, relevant_bits3), axis=1)

        for j in range(0, len(encoded_blocks4)):
            decoded4 = ldpc_encoder4.decode(received4[j, :, :])
            relevant_bits4 = decoded4[:, :n4-m4]
            final_message4 = np.concatenate((final_message4, relevant_bits4), axis=1)

        # Calculate the number of different bits for each case
        # Case 0
        print("final_message0: ", final_message0.shape)
        bit_errors0 = np.sum(message != final_message0)
        error_rate0 = bit_errors0 / x_info_bits
        error_rates0.append(error_rate0)
        print("Error rate0: ", error_rate0)

        # Case 1
        bit_errors1 = np.sum(message[0, :x1_info_bits] != final_message1[0,:])
        error_rate1 = bit_errors1 / x1_info_bits
        error_rates1.append(error_rate1)
        print("Error rate1: ", error_rate1)

        # Case 2
        bit_errors2 = np.sum(message[0, :x2_info_bits] != final_message2[0,:])
        error_rate2 = bit_errors2 / x2_info_bits
        error_rates2.append(error_rate2)
        print("Error rate2: ", error_rate2)

        # Case 3
        bit_errors3 = np.sum(message[0, :x3_info_bits] != final_message3[0,:])
        error_rate3 = bit_errors3 / x3_info_bits
        error_rates3.append(error_rate3)
        print("Error rate3: ", error_rate3)

        # Case 4
        bit_errors4 = np.sum(message[0, :x4_info_bits] != final_message4[0,:])
        error_rate4 = bit_errors4 / x4_info_bits
        error_rates4.append(error_rate4)
        print("Error rate4: ", error_rate4)


    # Plot the error rates
    plt.figure(figsize=(10, 6))
    plt.plot(p, error_rates0, 'r-o', label='Hamming(7, 4) Code')
    plt.plot(p, error_rates1, 'g-o', label='LDPC Code (n = 98)')
    plt.plot(p, error_rates2, 'b-o', label='LDPC Code (n = 203)')
    plt.plot(p, error_rates3, 'y-o', label='LDPC Code (n = 497)')
    plt.plot(p, error_rates4, 'm-o', label='LDPC Code (n = 1001)')
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

def main():
    simulate(x_info_bits=1000000, p=[0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])

if __name__ == "__main__":
    main()