"""Lab 04 A* foundation extended with Lab 07 logical feasibility."""
import heapq
import math
import random
from logic_engine import KnowledgeBase

# Match the original environment: Up increases y, Down decreases y.
MOVES = {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}


class GreedyGridAgent:
    """Retained starter interface for the original nonvisual simulator."""
    def sense_and_act(self, percept):
        return random.choice(list(MOVES))


class SearchAgent:
    def __init__(self):
        self.active_algo = 'AStar'
        self.plan = []
        self.kb = KnowledgeBase()
        self.kb.tell_rule(['TargetVisible', 'HasDust'], 'SafeToEngage')
        self.kb.tell_rule(['SafeToEngage', 'BloodseekerMissing'], 'Retreat')

    def manhattan_distance(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal):
        return math.hypot(pos[0] - goal[0], pos[1] - goal[1])

    def astar_search(self, start_pos, goal_pos, walls, grid_size,
                     heuristic_type='manhattan', tile_facts=None):
        """Return movement actions, [] if already there, or None if no safe path.

        grid_size is (width, height). Each orthogonal move costs one.
        An empty fact list does not prove safety: the lab only rejects Retreat.
        """
        tile_facts = {} if tile_facts is None else tile_facts
        start_pos, goal_pos = tuple(start_pos), tuple(goal_pos)
        walls = set(map(tuple, walls))
        width, height = grid_size
        if heuristic_type not in ('manhattan', 'euclidean'):
            raise ValueError('Use manhattan or euclidean')
        heuristic = (self.manhattan_distance if heuristic_type == 'manhattan'
                     else self.euclidean_distance)
        for pos in (start_pos, goal_pos):
            if pos in walls or not (0 <= pos[0] < width and 0 <= pos[1] < height):
                return None
        open_list = [(heuristic(start_pos, goal_pos), 0, start_pos, [])]
        reached_states = set()
        best_g = {start_pos: 0}
        while open_list:
            _, g_cost, current_pos, path_taken = heapq.heappop(open_list)
            if current_pos in reached_states:
                continue
            if current_pos == goal_pos:
                return path_taken
            reached_states.add(current_pos)
            for action, (dx, dy) in MOVES.items():
                successor = (current_pos[0] + dx, current_pos[1] + dy)
                # Physical reachability: bounds and walls.
                if not (0 <= successor[0] < width and 0 <= successor[1] < height):
                    continue
                if successor in walls or successor in reached_states:
                    continue
                # Logical feasibility: evaluate this tile independently.
                self.kb.clear_facts()
                for fact in tile_facts.get(successor, []):
                    self.kb.tell_fact(fact)
                self.kb.forward_chain()
                if 'Retreat' in self.kb.facts:
                    continue
                new_g = g_cost + 1
                if new_g < best_g.get(successor, float('inf')):
                    best_g[successor] = new_g
                    new_f = new_g + heuristic(successor, goal_pos)
                    heapq.heappush(open_list, (new_f, new_g, successor,
                                             path_taken + [action]))
        return None

    def sense_and_act(self, percept):
        position = tuple(percept['agent_pos'])
        # The starter's remaining_food is a COUNT. Preserve it and add a
        # separate food_positions percept containing the actual coordinates.
        food = sorted(map(tuple, percept['food_positions']),
                      key=lambda target: (self.manhattan_distance(position, target), target))
        walls = percept['walls']
        grid_size = percept['grid_size']
        tile_facts = {
            (1, 0): ['TargetVisible', 'HasDust', 'BloodseekerMissing'],
            (0, 1): ['TargetVisible', 'HasDust'],
        }
        # Replan every step so changes to food or the map do not leave a stale plan.
        self.plan = []
        for nearest_food in food:
            path = self.astar_search(position, nearest_food, walls, grid_size,
                                     tile_facts=tile_facts)
            if path is not None:
                self.plan = path
                return self.plan.pop(0) if self.plan else 'Stay'
        return 'Stay'  # Never fall back to random movement through unsafe tiles.
