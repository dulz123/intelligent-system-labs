"""Additional Lab 07 checks; run with python -m unittest test_integration -v."""
import unittest
from agent import SearchAgent, MOVES
from logic_engine import KnowledgeBase
from visual_grid_game import VisualGridHuntGame

UNSAFE = {(1, 0): ['TargetVisible', 'HasDust', 'BloodseekerMissing']}


class Lab07Tests(unittest.TestCase):
    def test_rule_order_and_cycles(self):
        kb = KnowledgeBase()
        kb.tell_rule(['B'], 'C')
        kb.tell_rule(['A'], 'B')
        kb.tell_rule(['C'], 'A')
        kb.tell_fact('A')
        kb.forward_chain()
        self.assertEqual(kb.facts, {'A', 'B', 'C'})
        kb.clear_facts()
        kb.forward_chain()
        self.assertEqual(kb.facts, set())
        self.assertEqual(len(kb.rules), 3)

    def test_missing_premise(self):
        a = SearchAgent()
        a.kb.tell_fact('TargetVisible')
        a.kb.tell_fact('BloodseekerMissing')
        a.kb.forward_chain()
        self.assertNotIn('Retreat', a.kb.facts)

    def test_detour_and_fact_isolation(self):
        a = SearchAgent()
        for heuristic in ('manhattan', 'euclidean'):
            path = a.astar_search((0, 0), (2, 0), set(), (3, 2), heuristic,
                                 tile_facts=UNSAFE)
            self.assertEqual(len(path), 4)
            pos = (0, 0)
            for action in path:
                dx, dy = MOVES[action]
                pos = (pos[0] + dx, pos[1] + dy)
                self.assertNotEqual(pos, (1, 0))
            self.assertEqual(pos, (2, 0))
        self.assertEqual(a.astar_search((0, 0), (2, 0), set(), (3, 2)),
                         ['Right', 'Right'])

    def test_no_path_and_zero_length(self):
        a = SearchAgent()
        self.assertIsNone(a.astar_search((0, 0), (2, 0), set(), (3, 1), tile_facts=UNSAFE))
        self.assertIsNone(a.astar_search((0, 0), (1, 0), {(1, 0)}, (2, 2)))
        self.assertEqual(a.astar_search((0, 0), (0, 0), set(), (2, 2)), [])
        self.assertEqual(a.manhattan_distance((0, 0), (3, 4)), 7)
        self.assertEqual(a.euclidean_distance((0, 0), (3, 4)), 5.0)

    def test_complete_environment_run(self):
        env = VisualGridHuntGame(6, 6, 0, 0, {(2, 2), (2, 3), (4, 3)})
        env.food_positions = {(2, 0), (4, 1), (5, 5)}
        agent = SearchAgent()
        while not env.is_done():
            env.execute_action(agent.sense_and_act(env.get_percept()))
            self.assertNotEqual(tuple(env.agent_pos), (1, 0))
        self.assertEqual(env.food_positions, set())
        self.assertEqual(env.score, 60)


if __name__ == '__main__':
    unittest.main()
