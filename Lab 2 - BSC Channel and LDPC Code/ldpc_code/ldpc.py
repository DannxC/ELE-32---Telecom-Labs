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

class LDPCDecoder:
    def __init__(self, n: int, dv: int, dc: int):   # n: codeword length, k: message length, dv: degree of variable nodes, dc: degree of check nodes        
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
        self.graph = Graph()
        self.generate_tanner_graph()

    # Generate Tanner graph
    # From idx 0 to n-1: variable nodes
    # From idx n to n+m-1: check nodes
    def generate_tanner_graph(self):
        # Add vertices for variable and check nodes
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
                idx = random.randint(0, len(check_nodes_copy)-1)
                current_check_node = check_nodes_copy[idx]

                self.graph.add_edge(variable_node, current_check_node)
                check_nodes_copy.pop(idx)

                check_nodes_count[current_check_node] += 1
                if check_nodes_count[current_check_node] == self.dc:
                    check_nodes.remove(current_check_node)
                    check_nodes_count.pop(current_check_node)


def main():
    # Initialize LDPC Decoder with parameters
    # n: number of variable nodes
    # dv: degree of each variable node
    # dc: degree of each check node
    decoder = LDPCDecoder(20, 3, 6)  # Example parameters
    decoder.graph.display()

if __name__ == "__main__":
    main()