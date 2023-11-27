import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
import math
from alive_progress import alive_bar
import cupy as cp
import networkx as nx

class NXImplementation:

    def heauristic(cell, goal):
        x1, y1 = cell
        x2, y2 = goal

        dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
        return dist
    
    def createGraph(point_stack_valid, depth_line, y_axis, start, destination):

        """
        Ths graphs needs to be from (0,0) untranslated, to (maxDepth, 0)
        This means, the y-axis needs to be (-1500, 1500) regardless, and the x-axis needs to be pruned
        If the entire map is translated, the start and end also need to be translated
        It suffices to move the graph entirely entirely along +ve y, or not at all
        """
        depth_line, y_axis = [], []
        for arr in point_stack_valid:
            depth_line.append(arr[0])
            y_axis.append(arr[1])

        depth_line = np.array(depth_line)
        y_axis = np.array(y_axis)

        translation_factor = [np.min(depth_line), -1500]   # for easier code, the entire map needs to be translated to origin
        rangeMax = [np.max(depth_line), 1500]

        print(translation_factor, rangeMax)

        gridGraph = nx.grid_2d_graph(int(rangeMax[0] - translation_factor[0]), int(rangeMax[1] - translation_factor[1]), False)
        point_stack_valid_translated = []

        for point in point_stack_valid:
            point_stack_valid_translated.append((point[0] - translation_factor[0], point[1] - translation_factor[1]))

        gridGraph.remove_nodes_from(point_stack_valid_translated)

        start = tuple(e1 - e2 for e1, e2 in zip(start, translation_factor))
        destination = tuple(e1 - e2 for e1, e2 in zip(destination, translation_factor))

        print(start, destination)

        assert destination in gridGraph, "Destination not present"
        assert start in gridGraph, "Start not present"

        optimal_path = nx.astar_path(gridGraph,
                    start,
                    destination,
                    NXImplementation.heauristic)    # the start and end need to be factored intelligently
        
        # optimal_path = nx.astar_path(gridGraph,
        #             (52, 748),
        #             (54, 792),
        #             NXImplementation.heauristic)    

        # this path needs to be inverse translated
        print("\n\n\n")

        print(optimal_path)

        for i in range(len(optimal_path)):
            optimal_path[i] = (optimal_path[i][0] + translation_factor[0], optimal_path[i][1] + translation_factor[1])

        print("reaches here")
        print(optimal_path)

        return optimal_path

    def createGraphXXX(point_stack_valid, depth_line, y_axis, start, destination):

        """
        Ths graphs needs to be from (0,0) untranslated, to (maxDepth, 0)
        This means, the y-axis needs to be (-1500, 1500) regardless, and the x-axis needs to be pruned
        If the entire map is translated, the start and end also need to be translated
        It suffices to move the graph entirely entirely along +ve y, or not at all
        For that matter, we can shift +1600 and be done with it
        """
        depth_line, y_axis = [], []
        for arr in point_stack_valid:
            depth_line.append(arr[0])
            y_axis.append(arr[1])

        depth_line = np.array(depth_line)
        y_axis = np.array(y_axis)

        translation_factor_y = 1500

        gridGraph = nx.grid_2d_graph(int(max(10,np.max(depth_line))+10), (int(max(translation_factor_y,np.max(y_axis) + translation_factor_y) + 10)), False)
        point_stack_valid_translated = []

        for point in point_stack_valid:
            point_stack_valid_translated.append((point[0], point[1] + translation_factor_y))

        gridGraph.remove_nodes_from(point_stack_valid_translated)

        start = (start[0], start[1] + translation_factor_y)
        destination = (destination[0], destination[1] + translation_factor_y)

        print(gridGraph.nodes)

        print(int(np.max(depth_line)+10), (int(np.max(y_axis) + translation_factor_y + 10)))
        print(start, destination)

        
        assert start in gridGraph, "Start not present"
        assert destination in gridGraph, "Destination not present"

        optimal_path = nx.astar_path(gridGraph,
                    start,
                    destination,
                    NXImplementation.heauristic)    # the start and end need to be factored intelligently
        
        # optimal_path = nx.astar_path(gridGraph,
        #             (52, 748),
        #             (54, 792),
        #             NXImplementation.heauristic)    

        # this path needs to be inverse translated
        print("\n\n\n")

        print(optimal_path)

        for i in range(len(optimal_path)):
            optimal_path[i] = (optimal_path[i][0], optimal_path[i][1] - translation_factor_y)

        print("reaches here")
        print(optimal_path)

        return optimal_path
    
class VanillaImplementation:
    def reconstruct_path(came_from, current):
        print("Entered reconstruct path, therefore some optimal path found")
        final_path = [current]
        while current in came_from:
            current = came_from[current]
            final_path.append(current)
        final_path = final_path[::-1]

        # final path is the collection of points of the graph
        return final_path

    def heauristic(cell, goal):
        x1, y1 = cell
        x2, y2 = goal

        dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
        return dist


    def A_star(graph, start, goal):
        MAXVAL = 100000000

        closed_set = []  # nodes already evaluated

        open_set = [start]  # nodes discovered but not yet evaluated

        came_from = {}  # most efficient path to reach from

        gscore = {}  # cost to get to that node from start

        for key in graph:
            gscore[key] = MAXVAL  # intialize cost for every node to inf

        gscore[start] = 0

        fscore = {}  # cost to get to goal from start node via that node

        for key in graph:
            fscore[key] = MAXVAL

        fscore[start] = VanillaImplementation.heauristic(start, goal)  # cost for start is only h(x)

        while open_set:
            min_val = MAXVAL*10  # find node in openset with lowest fscore value
            for node in open_set:
                if fscore[node] < min_val:
                    min_val = fscore[node]
                    min_node = node

            current = min_node  # set that node to current
            if current == goal:
                return VanillaImplementation.reconstruct_path(came_from, current)
            
            open_set.remove(current)  # remove node from set to be evaluated and
            closed_set.append(current)  # add it to set of evaluated nodes

            for neighbor in graph[current]:  # check neighbors of current node
                if neighbor in closed_set:  # ignore neighbor node if its already evaluated
                    continue
                if neighbor not in open_set:  # else add it to set of nodes to be evaluated
                    open_set.append(neighbor)

                # dist from start to neighbor through current
                tentative_gscore = gscore[current] + 1

                # not a better path to reach neighbor
                if tentative_gscore >= gscore[neighbor]:
                    continue
                came_from[neighbor] = current  # record the best path untill now
                gscore[neighbor] = tentative_gscore
                fscore[neighbor] = gscore[neighbor] + VanillaImplementation.heauristic(neighbor, goal)
        return False

    def createMap(point_stack_valid):

        depth_line, y_axis = [], []
        for arr in point_stack_valid:
            depth_line.append(arr[0])
            y_axis.append(arr[1])

        depth_line = cp.array(depth_line)
        y_axis = cp.array(y_axis)

        depth_line_min, depth_line_max = min(0, cp.min(depth_line)), cp.max(depth_line)
        y_axis_min, y_axis_max = min(0, cp.min(y_axis)), cp.max(y_axis)

        gridMap = defaultdict(list)

        for x in range(math.floor(depth_line_min), math.ceil(depth_line_max)):
            for y in range(math.floor(y_axis_min), math.ceil(y_axis_max)):
                # point_stack_valid = cp.array(point_stack_valid)
                if point_stack_valid.count((x,y)) == 0:
                    for dx, dy in [(-1, 0),
                                (1, 0),
                                (0, 1),
                                (0, -1),
                                (-1, -1),
                                (-1, 1),
                                (1, -1),
                                (1, 1),
                                (0, 0)]:
                        if depth_line_min <= x+dx < depth_line_max and y_axis_min <= y+dy < y_axis_max and point_stack_valid.count([x + dx, y + dy]) == 0:
                            if (x+dx, y+dy) not in gridMap[(x, y)]:
                                gridMap[(x, y)].append((x+dx, y+dy))

        # this creates all possible paths for the agent to move considering 8-cardinality
        print(gridMap)
        return gridMap

    def PathPlanner(point_stack_valid, start, destination):
        with alive_bar(200, bar = 'bubbles', spinner = 'notes2') as bar: 
            graph = VanillaImplementation.createMap(point_stack_valid)
            bar()

        #point stack valid will not contain the traversable points, graph will
        if not start in graph:
            print("Starting point is not a part of the valid map. \n")
            return

        elif not destination in graph:
            print("Destination is not a part of the valid map. \n")
            return
        
        shortest_route = VanillaImplementation.A_star(graph, start, destination)  

        return shortest_route