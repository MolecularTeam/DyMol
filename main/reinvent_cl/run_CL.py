import os
import sys
import numpy as np
from dacite import from_dict
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('/'.join(path_here.rstrip('/').split('/')[:-2]))
sys.path.append(path_here + "/diversity_filters")
from diversity_filters.filters import DiversityFilter
from diversity_filters.component_summary import DiversityFilterParameters, FinalSummary
from diversity_filters.conversions import Conversions
from main.optimizer import BaseOptimizer, Objdict
from utils import Variable, seq_to_smiles, unique
from model import RNN
from data_structs import Vocabulary, Experience
import torch
from evaluators.hypervolume import thermometer, generate_simplex
import yaml
from pymoo.util.ref_dirs import get_reference_directions


class REINVENTGame_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "reinvent_CL"

    def _optimize(self, oracle, config):
        self.oracle.assign_evaluator(oracle)
        self.obj_dim = len(oracle.name.split('+'))
        config = Objdict(config)
        path_here = os.path.dirname(os.path.realpath(__file__))
        restore_prior_from = os.path.join(path_here, 'data/Prior.ckpt')
        restore_agent_from = restore_prior_from
        voc = Vocabulary(init_from_file=os.path.join(path_here, "data/Voc"))

        Prior = RNN(voc)
        Agent = RNN(voc)
        diversity_config = from_dict(data_class=DiversityFilterParameters, data=config.diversity_filter)
        diversity_config.name = self.args.div_filter
        diversity_filter = DiversityFilter(diversity_config)


        ########## 추가 부분 ##########
        exp_dir = f'{config.save_path}'
        if self.args.do_save:
            do_save = True
            os.makedirs(exp_dir, exist_ok=True)
        else:
            do_save = False

        #############################
        ########## save_stuff ###########
        def save_stuff():
            with open(os.path.join(exp_dir, f"{self.args.timestamp}.ckpt"), "wb") as f:
                torch.save(
                    {
                        "args": self.args,
                        "config": config,
                        "model_state_dict": Agent.rnn.state_dict(),
                    },
                    f,
                )

        # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
        # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
        # to the CPU.
        if torch.cuda.is_available():
            Prior.rnn.load_state_dict(torch.load(os.path.join(path_here, 'data/Prior.ckpt')), strict=False)
            if self.args.load_pretrained:
                ckpt_path = os.path.join(exp_dir, self.args.load_pretrained + ".ckpt")
                ckpt = torch.load(ckpt_path)
                Agent.rnn.load_state_dict(ckpt["model_state_dict"], strict=False)
            else:
                Agent.rnn.load_state_dict(torch.load(restore_agent_from), strict=False)
        else:
            Prior.rnn.load_state_dict(
                torch.load(os.path.join(path_here, 'data/Prior.ckpt'), map_location=lambda storage, loc: storage), strict=False)
            Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage), strict=False)

        # We dont need gradients with respect to Prior
        for param in Prior.rnn.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=config['learning_rate'])

        # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
        # occur more often (which means the agent can get biased towards them). Using experience replay is
        # therefor not as theoretically sound as it is for value based RL, but it seems to work well.

        experience = Experience(
            voc)  # Class for prioritized experience replay that remembers the highest scored sequences seen and samples from them with probabilities relative to their scores.

        print("Model initialized, starting training...")

        step = 0
        patience = 0
        cl_count = 0
        cl_patience = 0
        check_phase = 0
        phase = 0
        init_cl = 0
        if self.args.update_order is not None:
            update_order = [int(i) for i in self.args.update_order.split(',')]
            log = "update order "
            for i, name in enumerate(self.oracle.evaluator.name_list):
                log += f"{name}: {update_order[i]} "
            print(log)
            update_order = np.array(update_order)
            phase = 1
            init_cl = 1
            check_phase = 1

        num_independents = 500
        independent_fighters = []


        def adaptation(experience: Experience, Agent, optimizer, num_adaptation=1):
            w_l = experience.weight_list
            f_w = experience.final_weight
            experience.final_weight = w_l

            for _ in range(num_adaptation):
                exp_seqs, exp_score, exp_prior_likelihood = experience.sample(len(experience))
                exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
                exp_augmented_likelihood = exp_prior_likelihood + config['sigma'] * exp_score
                exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
                loss = exp_loss
                agent_likelihood = exp_agent_likelihood
                loss = loss.mean()
                # Add regularizer that penalizes high likelihood for the entire sequence
                loss_p = - (1 / agent_likelihood).mean()
                loss += 5 * 1e3 * loss_p
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            experience.final_weight = f_w
            return

        def calculate_update_order(start_value, end_value):
            num_obj = start_value.shape[-1]
            update_order = [0] * num_obj

            # Calculate complexity (difference between end and start values)
            complexity = (end_value - start_value)
            print(complexity)

            # Identify objectives that exceed the threshold and should be updated last
            last_update_indices = [i for i, v in enumerate(start_value) if v > 0.25]

            # Sort the rest based on complexity, ignoring those that should be updated last
            sorted_indices = np.argsort(
                [complexity[i] if i not in last_update_indices else -np.inf for i in range(num_obj)])

            # Assign update order, ensuring those exceeding the threshold are updated last
            order = 0
            for i in sorted_indices:
                if i in last_update_indices:
                    update_order[i] = num_obj - len(last_update_indices)  # Set to one less than the total number of objectives
                else:
                    update_order[i] = order
                    order += 1

            return update_order

        while True:

            if len(self.oracle) > 100:
                self.sort_buffer()
                old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
            else:
                old_scores = 0

            if phase != check_phase:
                check_phase = phase
                cl_count = len(self.oracle)
                print("Phase shift: ", check_phase)
                adaptation(experience, Agent, optimizer, num_adaptation=1)


            # Sample from Agent

            seqs, agent_likelihood, entropy = Agent.sample(config['batch_size'])
            # Remove duplicates, ie only consider unique seqs
            unique_idxs = unique(seqs)
            seqs = seqs[unique_idxs]
            agent_likelihood = agent_likelihood[unique_idxs]
            # entropy = entropy[unique_idxs]
            uni_smiles = seq_to_smiles(seqs, voc)
            molecules, valid_indices = Conversions.smiles_to_mols_and_indices(uni_smiles)
            seqs, agent_likelihood = seqs[valid_indices], agent_likelihood[valid_indices]
            smiles = [uni_smiles[i] for i in valid_indices]
            if diversity_config.name != 'NoFilter' and init_cl == 1:
                smiles, scaffold_list, survive_idx = diversity_filter.filter_by_scaffold(smiles)
                seqs, agent_likelihood = seqs[survive_idx], agent_likelihood[survive_idx]

            # Get prior likelihood and score
            # prior_likelihood, _ = Prior.likelihood(Variable(seqs), cond_var[unique_idxs].cuda())
            prior_likelihood, _ = Prior.likelihood(Variable(seqs))
            smiles = seq_to_smiles(seqs, voc)
            # score = np.array(self.oracle(smiles))
            score, all_score = self.oracle(smiles, return_all=True)

            if diversity_config.name != 'NoFilter' and init_cl == 1:
                diversity_filter.add_with_filtered(FinalSummary(score, smiles, None), scaffold_list, step)
            # all_score = np.array(all_score)
            # score = (all_score * pref).sum(-1)
            all_score = np.array(all_score)

            log = "Average score | "
            for i, name in enumerate(self.oracle.evaluator.name_list):
                log += f"{name}: {np.mean(all_score, axis=0)[i].item():3f} "
            print(log)

            final_weight = np.array([1.]*self.obj_dim)
            final_weight /= final_weight.sum()
            experience.final_weight = final_weight
            CL_weight = final_weight * 1.2
            experience.weight_list = CL_weight

            if init_cl == 0:
                independent_fighters.append(all_score)
                if len(self.oracle) > num_independents:
                    init_cl += 1
                    start_fighter = np.concatenate([independent_fighters[0], independent_fighters[1]]).mean(0)
                    last_man_standing = np.concatenate([independent_fighters[-1], independent_fighters[-2]]).mean(0)
                    update_order = calculate_update_order(start_fighter, last_man_standing)

                    log = "update order "
                    for i, name in enumerate(self.oracle.evaluator.name_list):
                        log += f"{name}: {update_order[i]} "
                    print(log)
                    update_order = np.array(update_order)
                    phase += 1
            else:
                obj_flag = np.array([0.] * self.obj_dim)
                for cl_obj_idx in range(phase):
                    cur_flag = (update_order == cl_obj_idx)
                    obj_flag += cur_flag
                # production stage

                if phase == max(update_order)+2:  # last phase
                    CL_weight = obj_flag / obj_flag.sum() * 1.2
                    experience.weight_list = CL_weight
                else:
                    mean_score = np.mean(all_score[:, cur_flag.astype(bool)], axis=0)
                    if mean_score.mean() > 0.35 or len(self.oracle) - cl_count > 2500:
                        cl_patience += 1
                    # if obj_flag.sum() > 1:
                    obj_flag[cur_flag.astype(bool)] *= 1.5
                    CL_weight = obj_flag / obj_flag.sum()
                    weak_obj = (1e-08 < all_score.mean(axis=0)) * (all_score.mean(axis=0) < 0.4)
                    CL_weight[weak_obj] *= (0.4 + 0.15) / (all_score.mean(axis=0)[weak_obj] + 0.15)
                    experience.weight_list = CL_weight
                    if cl_patience > 1 and len(self.oracle) - cl_count > 250:
                        phase += 1
                        cl_patience = 0

            score_CL = np.sum(all_score * CL_weight, axis=-1)
            score_ex = all_score

            if self.finish:
                print('max oracle hit')
                break

            # Calculate augmented likelihood
            augmented_likelihood = prior_likelihood.float() + config['sigma'] * Variable(score_CL).float()
            loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

            # Experience Replay
            if len(experience) > config['experience_replay']:  # experience replay 24
                exp_seqs, exp_score, exp_prior_likelihood = experience.sample(config['experience_replay'] // 2)
                exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
                exp_augmented_likelihood = exp_prior_likelihood + config['sigma'] * exp_score
                exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
                loss = torch.cat((loss, exp_loss), 0)
                agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)
            if len(experience.pareto_memory) > config['experience_replay'] // 2:  # experience replay 24
                exp_seqs, exp_score, exp_prior_likelihood = experience.pareto_sample(config['experience_replay'] // 2)
                exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
                exp_augmented_likelihood = exp_prior_likelihood + config['sigma'] * exp_score
                exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
                loss = torch.cat((loss, exp_loss), 0)
                agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)


            # Then add new experience
            prior_likelihood = prior_likelihood.data.cpu().numpy()
            new_experience = zip(smiles, score_ex, prior_likelihood)
            new_experience_pare = zip(smiles, score_ex, prior_likelihood)
            experience.add_experience(new_experience)
            experience.add_pareto_experience(new_experience_pare, self.oracle.evaluator.pareto_smiles)

            # Calculate loss
            if config["AHC"] > 0.1:
                desc_idx = np.argsort(score)[::-1][:int(score.shape[0]*config["AHC"])].tolist()
                loss = loss[desc_idx]
                agent_likelihood = agent_likelihood[desc_idx]
            loss = loss.mean()
            # Add regularizer that penalizes high likelihood for the entire sequence
            loss_p = - (1 / agent_likelihood).mean()
            loss += 5 * 1e3 * loss_p

            # Calculate gradients and make an update to the network weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

        if do_save:
            save_stuff()
        print(f"timestamp: {self.args.timestamp}")
        print('Done.')

        def numpy_to_list(data):
            if isinstance(data, np.ndarray):
                return data.tolist()
            elif isinstance(data, np.float64):
                return data.item()
            elif isinstance(data, np.float32):
                return data.tolist()
            elif isinstance(data, dict):
                return {key: numpy_to_list(value) for key, value in data.items()}
            elif isinstance(data, tuple):
                return [numpy_to_list(x) for x in data]
            elif isinstance(data, list):
                return [numpy_to_list(x) for x in data]
            else:
                return data

        data = {"memory": experience.memory,
                "pareto_memory": experience.pareto_memory,
                "diversity_filter": diversity_filter.get_memory_as_dataframe().to_dict(orient='list')
                }

        converted_data = numpy_to_list(data)

        if not os.path.exists(self.args.memory_out_dir):
            os.mkdir(self.args.memory_out_dir)
        memory = os.path.join(self.args.memory_out_dir, self.args.timestamp + '.yaml')
        with open(memory, 'w') as f:
            yaml.dump(converted_data, f, sort_keys=False)

