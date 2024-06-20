import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import random   # For random number generation
from multiprocessing import Pool

# Graph with adjacency list representation
class Graph:
    def __init__(self):
        self.adj_list = {}   # {vertex: [neighbours]}

    def add_vertex(self, vertex):
        if vertex not in self.adj_list:
            self.adj_list[vertex] = []

    def add_edge(self, vertex1, vertex2, value=0):
        if vertex1 in self.adj_list and vertex2 in self.adj_list:
            self.adj_list[vertex1].append([vertex2, value])
            self.adj_list[vertex2].append([vertex1, value])  # Comment this line for a directed graph

    def att_edge_value(self, vertex1, vertex2, value):
        for edge in self.adj_list[vertex1]:
            if edge[0] == vertex2:
                edge[1] = value
        for edge in self.adj_list[vertex2]:
            if edge[0] == vertex1:
                edge[1] = value

    def display(self):
        for vertex in self.adj_list:
            print(f"{vertex}: {self.adj_list[vertex]}")

    def save_to_csv(self, filename):
        with open(filename, mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            for vertex in sorted(self.adj_list):
                # Lista apenas os primeiros elementos de cada sublista que são os índices dos c-nodes
                neighbors = [edge[0] for edge in self.adj_list[vertex]]
                # Escreve no arquivo o vértice atual seguido por seus vizinhos
                csv_writer.writerow(neighbors)


# Contains the encode and decode functions for LDPC codes (this case uses the LLR algorithm for decoding)
class LDPCEncoderWithLLR:
    # Initialize LDPC Encoder with parameters
    # n: number of variable nodes (or codeword length)
    # dv: degree of each variable node
    # dc: degree of each check node
    def __init__(self, n: int, dv: int, dc: int):
        # Calculate number of check nodes and check if it is integer
        self.m: int = n * dv // dc  # number of check nodes
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

        # Define the type of modulation (Here we are using BPSK with symbols +1 for bit=0 and -1 for bit=1)
        self.Eb = 1
        self.symbol0 = 1
        self.symbol1 = -1

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

    # Encode function (encode bits using generator matrix G)
    # message.shape should be (1, n-m)
    def encode(self, message):
        return message @ self.G % 2 # Return shape should be (1, n)
    
    # Convert bits to BPSK symbols using NumPy's vectorized operations
    # +1 for bit 0 and -1 for bit 1
    # encoded_message.shape should be (1, n)
    def encode_message_to_symbols(self, encoded_message):
        return np.where(encoded_message == 0, self.symbol0, self.symbol1)    # Returned shape should be (1, n)

    # Decode function
    # received is a vector that represents a group of self.n symbols
    # received.shape should be (1, n)
    def decode(self, received_llr):
        # Verifica se received_llr tem a forma esperada
        if received_llr.shape != (1, self.n):
            raise ValueError("received_llr deve ter a forma (1, n)")
    
        # Set each value on edges for 0
        for v_node in range(self.n):
            for edge in self.graph.adj_list[v_node]:
                self.graph.att_edge_value(v_node, edge[0], 0)

        # Initialize the message to be decoded
        decoded_llr = received_llr.copy()

        max_iterations = 0
        while max_iterations <= 10:
            # Calculate v-nodes
            for v_node in range(self.n):
                # Calculate the sum of all incoming messages to the v-node
                sum_incoming = received_llr[0, v_node] + sum([edge[1] for edge in self.graph.adj_list[v_node]])
                # Update the message in the graph for each edge
                for edge in self.graph.adj_list[v_node]:
                    self.graph.att_edge_value(v_node, edge[0], sum_incoming - edge[1])


            # Verify Grey's condition for each c-node
            stop_algorithm = True
            for c_node in range(self.n, self.n + self.m):
                # Calculate the multiplication of all signs of incoming edge values
                sign = 1
                for edge in self.graph.adj_list[c_node]:
                    sign *= np.sign(edge[1])

                # If the multiplication is negative, update the message in the graph for each edge
                # Otherwise, stop the algorithm
                if sign == -1:
                    stop_algorithm = False
                    break

            if stop_algorithm:
                break   # Stop the algorithm if Grey's condition is satisfied


            # Calculate c-nodes
            for c_node in range(self.n, self.n + self.m):
                # Calculate the multiplication of all signs of incoming edge values
                sign = 1
                for edge in self.graph.adj_list[c_node]:
                    sign *= np.sign(edge[1])
                

                # Update the value of each edge in the graph
                # Find the two smallest values in the absolute value of the edges
                (min1, min2) = LDPCEncoderWithLLR.find_two_smallest([abs(edge[1]) for edge in self.graph.adj_list[c_node]])
                
                # Update the value of each edge in the graph
                for edge in self.graph.adj_list[c_node]:
                    actual_sign = sign * np.sign(edge[1])  # only consider the sign of the other edges

                    # Att the value of the edge for the minimum value of the other edges
                    if edge[1] == min1:
                        self.graph.att_edge_value(c_node, edge[0], actual_sign * min2)
                    else:
                        self.graph.att_edge_value(c_node, edge[0], actual_sign * min1)

            max_iterations += 1


        # Calculate the decoded symbols
        for v_node in range(self.n):
            decoded_llr[0, v_node] += sum([edge[1] for edge in self.graph.adj_list[v_node]])

        # print("decoded_llr.shape: ", decoded_llr.shape)
        # print(decoded_llr[0, :10])

        # Calculate the final message (bits) from the decoded symbols
        decoded_message = np.empty((1, self.n), dtype=int)
        for i in range(self.n):
            decoded_message[0, i] = 0 if decoded_llr[0, i] >= 0 else 1

        # print("decoded_message.shape: ", decoded_message.shape)
        # print(decoded_message[0, :10])
        return decoded_message

    @staticmethod
    # Auxiliary function to find the two smallest numbers in a list
    def find_two_smallest(numbers):
        if len(numbers) < 2:
            return None  # Retorno None se não houver pelo menos dois números

        # Inicializa o primeiro e segundo menores com valores máximos
        if numbers[0] < numbers[1]:
            min1, min2 = numbers[0], numbers[1]
        else:
            min1, min2 = numbers[1], numbers[0]

        # Itera sobre o array a partir do terceiro elemento
        for num in numbers[2:]:
            if num < min1:
                min2 = min1  # Atualiza o segundo menor
                min1 = num   # Atualiza o menor
            elif num < min2:
                min2 = num   # Atualiza apenas o segundo menor

        return min1, min2


# Contains the encode and decode functions for LDPC codes (this case uses the Bit-Flip algorithm for decoding)
class LDPCEncoderWithBitFlip:
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
                node_list = [node[0] for node in self.graph.adj_list[i + self.n]] # Extract node indices
                parity_checks[i] = (sum(decoded[0, node] for node in node_list) % 2) == 0

                if not parity_checks[i]:
                    for edge in self.graph.adj_list[i + self.n]:
                        node = edge[0]
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


# Gaussian Channel class
class GaussianChannel:
    # Initialize Gaussian Channel with parameters
    # Eb_N0_dB: Energy per bit to noise power spectral density ratio in dB
    def __init__(self, Eb_N0_dB, Eb):
        self.Eb_N0_dB = Eb_N0_dB                    # Energy per bit to noise power spectral density ratio
        self.Eb_N0 = 10**(Eb_N0_dB / 10)            # Convert dB to linear scale (Eb/N0 = 10^(Eb/N0_dB/10))
        self.Eb = Eb                                # Energy per bit
        self.N0 = self.Eb / self.Eb_N0              # Noise power spectral density (N0 = Eb / Eb/N0)
        self.sigma = np.sqrt(self.N0 / 2)           # Standard deviation of noise (sigma = sqrt(1 / (2 * Eb/N0)))
    
    # Transmit function
    # encoded_symbols is a vector that represents a group of self.n symbols
    # encoded_symbols.shape = (1, n)
    def transmit(self, encoded_symbols):
        noise = np.random.normal(0, self.sigma, encoded_symbols.shape)      # Generate Gaussian noise with mean 0 and standard deviation sigma
        received_symbols = encoded_symbols + noise                          # Add noise to the codeword to get the received signal
        return received_symbols

    # Calculate LLR function
    # received is a vector that represents a group of self.n bits
    def calculate_llr(self, received):
        llr = 2 * received / self.sigma**2  # Log-likelihood ratio
        return llr


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


# Save a matplotlib plot with a unique filename in the specified directory
def save_plot_with_unique_name(x_info_bits, base_filename="ldpc_gaussian_channel", extension=".png", directory="."):
    """
    Save a matplotlib plot with a unique filename in the specified directory.
    Filename dynamically scales the number of information bits to reflect the magnitude in thousands (k).
    """
    # Determinar quantas vezes o número pode ser dividido por 1000
    count_k = 0
    scaled_bits = x_info_bits
    while scaled_bits >= 1000:
        scaled_bits /= 1000
        count_k += 1

    # Criar o sufixo com base na quantidade de "k"s
    if count_k > 0:
        scaled_bits = int(scaled_bits)  # Transforma em inteiro se a divisão foi realizada
        suffix = f"{scaled_bits}" + "k" * count_k
    else:
        suffix = str(x_info_bits)  # Usar o número original se não houver divisão por mil

    formatted_filename = f"{suffix}_bits_{base_filename}{extension}"
    full_path = os.path.join(directory, formatted_filename)

    # Verifica se o arquivo já existe e ajusta o nome se necessário
    counter = 1
    while os.path.exists(full_path):
        formatted_filename = f"{suffix}_bits_{base_filename}_{counter}{extension}"
        full_path = os.path.join(directory, formatted_filename)
        counter += 1

    # Salva a figura com o nome que estiver disponível
    plt.savefig(full_path)
    print(f"Plot saved as: {full_path}")


# Generate a 0's x-bits message
def generateInformationBits(x):
    # return np.random.randint(0, 2, x).astype(int).reshape(1,x)
    return np.zeros(x, dtype=int).reshape(1, x)


# Simulate and Compare methods for different values of Eb_N0_db using parallelism
def simulate_single1(args):
    message, Eb_N0_dB, ldpc_encoder1, x1_info_bits, encoded_symbols1, n1, m1 = args

    # Declaring the necessary variables to store the final message and error rates
    error_rates1 = []
    
    # Transmit each block through the channel for different values of p
    print("\nTRANSMISSION THROUGH GAUSSIAN CHANNEL\n")

    print("\n\tEb/N0 (dB) = ", Eb_N0_dB, "\n")
    channel1 = GaussianChannel(Eb_N0_dB, ldpc_encoder1.Eb)

    # Initialize the necessary np arrays to store the received symbols
    received_symbols1 = np.empty((1, 0), dtype=int)
    final_message1 = np.empty((1, 0), dtype=int)

    # Case 1 - Transmit the encoded symbols 1 through the channel (all at once)
    received_symbols1 = channel1.transmit(encoded_symbols1)
    received_llr1 = channel1.calculate_llr(received_symbols1)

    # print("received_llr1.shape: ", received_llr1.shape)
    print(received_llr1[0, :, :10])

    # Decode the received blocks for each case
    for j in range(0, encoded_symbols1.shape[0]):
        decoded_message1 = ldpc_encoder1.decode(received_llr1[j, :, :])     # input has shape (1, n1), and output too (but now represent bits)
        relevant_bits1 = decoded_message1[:, :n1-m1]
        # print("\ndecoded_message1.shape: ", decoded_message1.shape)
        # print("relevant_bits1.shape: ", relevant_bits1.shape)
        final_message1 = np.concatenate((final_message1, relevant_bits1), axis=1)

    # print("\n\nfinal_message1.shape: ", final_message1.shape)
    # print(final_message1[0, :10])

    # Calculate the number of different bits for each case
    # Case 1
    bit_errors1 = np.sum(message[0, :x1_info_bits] != final_message1[0,:])
    error_rate1 = bit_errors1 / x1_info_bits
    error_rates1.append(error_rate1)
    print("Error rate1: ", error_rate1)

    return error_rate1


# Simulate and Compare methods for different values of p using parallelism
def simulate_single2(args):
    message, Eb_N0_dB, ldpc_encoder1, ldpc_encoder2, x2_info_bits, encoded_blocks2, n2, m2 = args
    
    # Declaring the necessary variables to store the final message and error rates
    error_rates2 = []
    
    # Transmit each block through the channel for different values of p
    print("\nTRANSMISSION THROUGH GAUSSIAN CHANNEL (simulating BSC here!)\n")

    print("\n\tInstead \"p\", quivalent Eb/N0 (dB) = ", Eb_N0_dB, "\n")
    channel1 = GaussianChannel(Eb_N0_dB, ldpc_encoder1.Eb)
    # channel2 = BSC(p)  # BSC channel... dont need here because we are simulating it with the gaussian channel!

    # Initialize the necessary np arrays to store the received symbols
    received_symbols1 = np.empty((1, 0), dtype=int)
    final_message2 = np.empty((1, 0), dtype=int)

    # Case 2 - Transmit the encoded symbols 1 through the channel (all at once)
    encoded_symbols1 = ldpc_encoder1.encode_message_to_symbols(encoded_blocks2)
    received_symbols1 = channel1.transmit(encoded_symbols1)

    # Converting received symbols to bits, because we are simulating a BSC (and its inputs are bits)
    # Deciding with a limiar on Zero (0) to convert to 0 or 1... rth = 0 here
    received2 = np.where(received_symbols1 >= 0, 0, 1)  # BPSK modulation: +1 for 0 and -1 for 1
    # print("received2.shape: ", received2.shape)
    # print(received2[:10])

    for j in range(0, len(encoded_blocks2)):
        decoded2 = ldpc_encoder2.decode(received2[j, :, :])
        relevant_bits2 = decoded2[:, :n2-m2]
        final_message2 = np.concatenate((final_message2, relevant_bits2), axis=1)

    bit_errors2 = np.sum(message[0, :x2_info_bits] != final_message2[0,:])
    error_rate2 = bit_errors2 / x2_info_bits
    error_rates2.append(error_rate2)
    print("Error rate2: ", error_rate2)
    
    return error_rate2


# Simulate and Compare methods for different values of p using parallelism
def simulate_parallel(x_info_bits, Ei_N0_dBs, process_count):
    
    # Generate message with x_info_bits
    message = generateInformationBits(x_info_bits)
    print("message.shape: ", message.shape)


    # CASE 1: LDPC with LLR algorithm and with n = 1001
    n1 = 1001               # codeword length (number of v-nodes)
    m1 = n1 * 3 // 7        # number of check nodes (nuber of c-nodes)
    Eb_N0_dBs = list(map(lambda x: x * (7/10), Ei_N0_dBs))
    ldpc_encoder1 = LDPCEncoderWithLLR(n1, 3, 7)   # n, dv, dc
    # ldpc_encoder1.graph.display()
    # ldpc_encoder1.graph.save_to_csv("tanner_graph1.csv")
    x1_info_bits = message.shape[1]-(message.shape[1] % (n1-m1))
    encoded_blocks1 = ldpc_encoder1.encode([message[:, i:i+n1-m1] for i in range(0, x1_info_bits, n1-m1)])
    encoded_symbols1 = ldpc_encoder1.encode_message_to_symbols(encoded_blocks1)
    print("encoded_symbols1.shape: ", encoded_symbols1.shape)


    # CASE 2: LDPC with bit-flip algorithm with n = 1001
    n2 = 1001
    m2 = n2 * 3 // 7
    #p = something -- Would be the probability of bit flip... but we are going to use the gaussian channel, because it would be the same as using BSC with the correct p for each Eb_N0_dB... it is easier to use the gaussian channel
    ldpc_encoder2 = LDPCEncoderWithBitFlip(n2, 3, 7)
    # ldpc_encoder2.graph.display()
    x2_info_bits = message.shape[1]-(message.shape[1] % (n2-m2))
    encoded_blocks2 = ldpc_encoder2.encode([message[:, i:i+n2-m2] for i in range(0, x2_info_bits, n2-m2)])
    print("encoded_blocks2.shape: ", encoded_blocks2.shape)




    # SIMULATE CASE 1
    # Create a pool of processes (parallelism)
    with Pool(process_count) as pool:
        # Create a partial function that all processes can use
        results = pool.map(simulate_single1, [(message, Eb_N0_dB, ldpc_encoder1, x1_info_bits, encoded_symbols1, n1, m1) for Eb_N0_dB in Eb_N0_dBs])
    
    error_rates1 = [result for result in results]   # here assuming only one result per process


    # CASE 2
    # SIMULATE CASE 2
    with Pool(process_count) as pool:
        # Create a partial function that all processes can use
        results = pool.map(simulate_single2, [(message, Eb_N0_dB, ldpc_encoder1, ldpc_encoder2, x2_info_bits, encoded_blocks2, n2, m2) for Eb_N0_dB in Eb_N0_dBs])

    error_rates2 = [result for result in results]   # here assuming only one result per process



    # Plot the error rates
    plt.figure(figsize=(10, 6))

    plt.plot(Ei_N0_dBs, error_rates1, 'g-o', label='LDPC Code (n = 1001) - LLR ')
    plt.plot(Ei_N0_dBs, error_rates2, 'm-o', label='LDPC Code (n = 1001) - Bit Flip ')
    plt.scatter(Ei_N0_dBs, error_rates1, color='red')  # mark each point
    plt.scatter(Ei_N0_dBs, error_rates2, color='blue')  # Mark each point for Case 2

    plt.xlabel('Ei/N0 (dB)')
    plt.ylabel('Error Rate')
    plt.title('Error Rate vs. Ei/N0')

    plt.xticks(Ei_N0_dBs, labels=[f"{x:.2f}" for x in Ei_N0_dBs])
    plt.xlim(min(Ei_N0_dBs), max(Ei_N0_dBs))
    plt.yscale('log')
    plt.yticks([1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6], labels=['1e0', '1e-1', '1e-2', '1e-3', '1e-4', '1e-5', '1e-6'])
    plt.ylim(1e-6, 1e0)
    
    plt.legend(title="Legend", title_fontsize='13', fontsize='11')  # Add a title to the legend for clarity
    plt.grid(True)

    save_plot_with_unique_name(x_info_bits)  # Altere conforme necessário para o seu caso
    plt.show()


def main():
    simulate_parallel(x_info_bits=100000, Ei_N0_dBs=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], process_count=8)


if __name__ == "__main__":
    main()