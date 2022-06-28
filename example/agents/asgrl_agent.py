from learning_agents.hierarchical_diversity_rl.hierarchical_diversity_rl_curriculum import Curriculum_Hierarchical_Diversity_RL


class Household_ASGRL(Curriculum_Hierarchical_Diversity_RL):
    def __init__(self, env, config, logger=None):
        super().__init__(env, config, logger=logger)

    def check_skill_success(self, state_info):
        curr_subgoal_finished = False
        other_satisfied_subgoals = set()
        if state_info is None:
            return False, set()
        if state_info.key_picked_first_door and 'key_picked_first_door' in self.agents_low:
            if self.curr_subgoal == 'key_picked_first_door':
                curr_subgoal_finished = True
            else:
                other_satisfied_subgoals.add('key_picked_first_door')
        if state_info.door_0_unlocked and 'door_0_unlocked' in self.agents_low:
            if self.curr_subgoal == 'door_0_unlocked':
                curr_subgoal_finished = True
            else:
                other_satisfied_subgoals.add('door_0_unlocked')
        if state_info.key_picked_second_door and 'key_picked_second_door' in self.agents_low:
            if self.curr_subgoal == 'key_picked_second_door':
                curr_subgoal_finished = True
            else:
                other_satisfied_subgoals.add('key_picked_second_door')
        if state_info.door_1_unlocked and 'door_1_unlocked' in self.agents_low:
            if self.curr_subgoal == 'door_1_unlocked':
                curr_subgoal_finished = True
            else:
                other_satisfied_subgoals.add('door_1_unlocked')
        if state_info.is_charged and 'is_charged' in self.agents_low:
            if self.curr_subgoal == 'is_charged':
                curr_subgoal_finished = True
            else:
                other_satisfied_subgoals.add('is_charged')
        if state_info.visited_room_7 and 'visited_room_7' in self.agents_low:
            if self.curr_subgoal == 'visited_room_7':
                curr_subgoal_finished = True
            else:
                other_satisfied_subgoals.add('visited_room_7')
        if state_info.at_destination and 'at_destination' in self.agents_low:
            if self.curr_subgoal == 'at_destination':
                curr_subgoal_finished = True
            else:
                other_satisfied_subgoals.add('at_destination')
        return curr_subgoal_finished, other_satisfied_subgoals

