import os
import pickle
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Any, List, NoReturn
from deep_sprl.teachers.util import RewardEstimatorGP
from deep_sprl.teachers.spl.assignment_solver import AssignmentSolver
from deep_sprl.teachers.abstract_teacher import AbstractTeacher
from deep_sprl.teachers.spl.wasserstein_interpolation import WassersteinInterpolation
from collections import deque
import math

import torch

class CurrOT(AbstractTeacher):

    def __init__(self, context_bounds, init_samples, target_sampler, perf_lb, epsilon, episodes_per_update,
                 callback=None, model=None, wait_until_threshold=False, wb_max_reuse=1):
        if model is None:
            self.model = RewardEstimatorGP()
        else:
            self.model = model

        self.smooth = True
        # Create an array if we use the same number of bins per dimension
        self.context_bounds = context_bounds
        self.threshold_reached = not wait_until_threshold
        self.teacher = WassersteinInterpolation(init_samples, target_sampler, perf_lb, epsilon, callback=callback)
        self.success_buffer = WassersteinSuccessBuffer(perf_lb, init_samples.shape[0], episodes_per_update, epsilon,
                                                       context_bounds=context_bounds, max_reuse=wb_max_reuse)
        self.fail_context_buffer = []
        self.fail_return_buffer = []
        self.sampler = UniformSampler(self.context_bounds)
        self.dist_queue = deque(maxlen=7)
        self.local_minima = False

        if self.smooth:
            # Added for randomized smoothing: initialize smoothing parameters
            self.sigma_0 = 0.1 * np.linalg.norm(context_bounds[1] - context_bounds[0])  # Initial variance based on context space scale
            self.sigma_t = self.sigma_0  # Current variance, updated per iteration
            self.monte_carlo_samples = 20  # Number of Monte Carlo samples (S) for smoothing
            self.iteration = 0  # Track iteration for sigma_t decay

    def on_rollout_end(self, context, ret):
        self.sampler.update(context, ret)

    def update_distribution(self, contexts, returns):
        fail_contexts, fail_returns = self.success_buffer.update(contexts, returns,
                                                                 self.teacher.target_sampler(
                                                                     self.teacher.current_samples.shape[0]))

        if self.threshold_reached:
            self.fail_context_buffer.extend(fail_contexts)
            self.fail_context_buffer = self.fail_context_buffer[-self.teacher.n_samples:]
            self.fail_return_buffer.extend(fail_returns)
            self.fail_return_buffer = self.fail_return_buffer[-self.teacher.n_samples:]

        success_contexts, success_returns = self.success_buffer.read_train()
        if len(self.fail_context_buffer) == 0:
            train_contexts = success_contexts
            train_returns = success_returns
        else:
            train_contexts = np.concatenate((np.stack(self.fail_context_buffer, axis=0), success_contexts), axis=0)
            train_returns = np.concatenate((np.stack(self.fail_return_buffer, axis=0), success_returns), axis=0)
        self.model.update_model(train_contexts, train_returns)

        if self.smooth:
            print("Updating sampling distribution with smoothed performance")
            current_samples = torch.tensor(self.teacher.current_samples, dtype=torch.float32)  # Convert to tensor
            smoothed_perf = 0.0
            for _ in range(self.monte_carlo_samples):
                # Perturb samples with Gaussian noise
                noise = torch.normal(mean=0.0, std=self.sigma_t, size=current_samples.shape)
                perturbed_samples = current_samples + noise
                # Clip to context bounds
                perturbed_samples = torch.clamp(perturbed_samples,
                                            min=torch.tensor(self.context_bounds[0], dtype=torch.float32),
                                            max=torch.tensor(self.context_bounds[1], dtype=torch.float32))
                # Compute performance using GP model
                perf = self.model(perturbed_samples.numpy())  # Convert back to numpy for model
                smoothed_perf += perf / self.monte_carlo_samples
            avg_perf = smoothed_perf  # Use smoothed performance for threshold check
        else:
            avg_perf = self.model(self.teacher.current_samples)

        original_samples = self.teacher.current_samples.copy()
        if self.threshold_reached or self.model(self.teacher.current_samples) >= self.teacher.perf_lb:
            self.threshold_reached = True
            if self.smooth:
                perturbed_samples_list = []
                for _ in range(5):
                    noise = torch.normal(mean=0.0, std=self.sigma_t, size=current_samples.shape)
                    perturbed_samples = current_samples + noise
                    perturbed_samples = torch.clamp(perturbed_samples,
                                                min=torch.tensor(self.context_bounds[0], dtype=torch.float32),
                                                max=torch.tensor(self.context_bounds[1], dtype=torch.float32))
                # Average samples for update (approximates smoothed gradient)
                # smoothed_samples = perturbed_samples_list[np.random.randint(len(perturbed_samples_list))]
                # smoothed_samples = np.mean(perturbed_samples_list, axis=0)
                # # Ensure smoothed constraint satisfaction via projection
                    # smoothed_perf = self.model(perturbed_samples.numpy())
                    # print(smoothed_perf, self.teacher.perf_lb)
                    # if smoothed_perf < self.teacher.perf_lb:
                    #     # Simple projection: scale back toward original samples
                    #     alpha = 0.7  # Projection step size
                    #     perturbed_samples = alpha * perturbed_samples + (1 - alpha) * self.teacher.current_samples
                    #     # Re-check and iterate if needed (basic projection for minimal changes)
                    #     for _ in range(2):  # Limited iterations to avoid complexity
                    #         smoothed_perf = self.model(perturbed_samples.numpy())
                    #         if smoothed_perf >= self.teacher.perf_lb:
                    #             break
                    #         print("Reverting to original samples due to low performance")
                    #         perturbed_samples = alpha * perturbed_samples + (1 - alpha) * self.teacher.current_samples
                    perturbed_samples_list.append(perturbed_samples.numpy())
                self.teacher.current_samples = np.mean(perturbed_samples_list, axis=0)  # Update with smoothed samples
            self.teacher.update_distribution_perturbed(self.model, perturbed_samples_list, self.success_buffer.read_update())
        else:
            print("Not updating sampling distribution, as performance threshold not met: %.3e vs %.3e" % (
                self.model(self.teacher.current_samples), self.teacher.perf_lb))


        smoothed_perf = self.model(self.teacher.current_samples))
        print(smoothed_perf, self.teacher.perf_lb)
        if smoothed_perf < self.teacher.perf_lb:
            # Simple projection: scale back toward original samples
            alpha = 0.7  # Projection step size
            self.teacher.current_samples = alpha * self.teacher.current_samples + (1 - alpha) * original_samples
            # Re-check and iterate if needed (basic projection for minimal changes)
            for _ in range(1):  # Limited iterations to avoid complexity
                smoothed_perf = self.model(self.teacher.current_samples)
                if smoothed_perf >= self.teacher.perf_lb:
                    break
                print("Reverting to original samples due to low performance")
                self.teacher.current_samples = alpha * self.teacher.current_samples + (1 - alpha) * self.teacher.current_samples

        if self.smooth:
            self.iteration += 1
            # self.sigma_t = self.sigma_0 / ((self.iteration + 1)**0.7)  # Decay variance
            average = sum(self.dist_queue) / len(self.dist_queue) if len(self.dist_queue) > 0 else 0
            variance = math.sqrt(sum((x - average) ** 2 for x in self.dist_queue) / len(self.dist_queue)) if len(self.dist_queue) > 4 else 0
            if self.iteration > 350:
                self.sigma_t = self.sigma_0 / ((self.iteration + 1)**0.7)
                pass
            else:
                if self.local_minima:
                    if abs(self.teacher.wass_dist - average) > 2 * variance:
                        self.sigma_t = self.sigma_0 / ((self.iteration + 1)**0.7)
                        self.dist_queue.clear()
                        self.local_minima=False
                    else:
                        print("Local minima detected, increasing sigma_t")
                        self.sigma_t = self.sigma_t * 1.5
                        self.dist_queue.append(self.teacher.wass_dist)
                else:
                    self.sigma_t = self.sigma_0 / ((self.iteration + 1)**0.7)  # Decay variance
                    if variance < 0.1 and abs(self.teacher.wass_dist - average) < 1.5*variance:
                        self.sigma_t = self.sigma_t * 1.5
                        self.local_minima = True
                    self.dist_queue.append(self.teacher.wass_dist)

    def sample(self):
        sample = self.sampler(self.teacher.current_samples)
        return np.clip(sample, self.context_bounds[0], self.context_bounds[1])

    def save(self, path):
        self.model.save(os.path.join(path, "teacher_model.pkl"))
        self.teacher.save(path)
        self.success_buffer.save(path)
        self.sampler.save(path)

    def load(self, path):
        self.model.load(os.path.join(path, "teacher_model.pkl"))
        self.teacher.load(path)
        self.success_buffer.load(path)
        self.sampler.load(path)


class AbstractSuccessBuffer(ABC):

    def __init__(self, delta: float, n: int, epsilon: float, context_bounds: Tuple[np.ndarray, np.ndarray]):
        context_exts = context_bounds[1] - context_bounds[0]
        self.delta_stds = context_exts / 4
        self.min_stds = 0.005 * epsilon * np.ones(len(context_bounds[0]))
        self.context_bounds = context_bounds
        self.delta = delta
        self.max_size = n
        self.contexts = np.zeros((1, len(context_bounds[0])))
        self.returns = np.array([-np.inf])
        self.delta_reached = False
        self.min_ret = None

    @abstractmethod
    def update_delta_not_reached(self, new_contexts: np.ndarray, new_returns: np.ndarray,
                                 current_samples: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, List[bool]]:
        pass

    @abstractmethod
    def update_delta_reached(self, new_contexts: np.ndarray, new_returns: np.ndarray, current_samples: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, List[bool]]:
        pass

    def update(self, contexts, returns, current_samples):
        assert contexts.shape[0] < self.max_size

        if self.min_ret is None:
            self.min_ret = np.min(returns)

        if not self.delta_reached:
            self.delta_reached, self.contexts, self.returns, mask = self.update_delta_not_reached(contexts, returns,
                                                                                                  current_samples)
        else:
            self.contexts, self.returns, mask = self.update_delta_reached(contexts, returns, current_samples)

        return contexts[mask, :], returns[mask]

    def read_train(self):
        return self.contexts.copy(), self.returns.copy()

    def read_update(self):
        # Compute the Gaussian search noise that we add to the samples
        var_scales = np.clip(self.delta - self.returns, 0., np.inf) / (self.delta - self.min_ret)
        stds = self.min_stds[None, :] + var_scales[:, None] * self.delta_stds[None, :]

        # If we did not yet reach the desired threshold we enforce exploration by scaling the exploration noise w.r.t.
        # the distance to the desired threshold value
        if not self.delta_reached:
            offset = self.returns.shape[0] // 2
            sub_returns = self.returns[offset:]
            sub_contexts = self.contexts[offset:, :]
            sub_stds = stds[offset:, :]

            # Do a resampling based on the achieved rewards (favouring higher rewards to be resampled)
            probs = sub_returns - self.returns[offset - 1]
            norm = np.sum(probs)
            if norm == 0:
                probs = np.ones(sub_returns.shape[0]) / sub_returns.shape[0]
            else:
                probs = probs / norm

            sample_idxs = np.random.choice(sub_returns.shape[0], self.max_size, p=probs)
            sampled_contexts = sub_contexts[sample_idxs, :]
            sampled_stds = sub_stds[sample_idxs, :]
        else:
            to_fill = self.max_size - self.returns.shape[0]
            add_idxs = np.random.choice(self.returns.shape[0], to_fill)
            sampled_contexts = np.concatenate((self.contexts, self.contexts[add_idxs, :]), axis=0)
            sampled_stds = np.concatenate((stds, stds[add_idxs, :]), axis=0)

        contexts = sampled_contexts + np.random.normal(0, sampled_stds, size=(self.max_size, self.contexts.shape[1]))
        invalid = np.any(np.logical_or(contexts < self.context_bounds[0][None, :],
                                       contexts > self.context_bounds[1][None, :]), axis=-1)
        count = 0
        while np.any(invalid) and count < 10:
            new_noise = np.random.normal(0, sampled_stds[invalid, :], size=(np.sum(invalid), self.contexts.shape[1]))
            contexts[invalid, :] = sampled_contexts[invalid, :] + new_noise
            invalid = np.any(np.logical_or(contexts < self.context_bounds[0][None, :],
                                           contexts > self.context_bounds[1][None, :]), axis=-1)
            count += 1

        return np.clip(contexts, self.context_bounds[0], self.context_bounds[1])

    def get_data(self) -> Any:
        return None

    def set_data(self, data: Any) -> NoReturn:
        pass

    def save(self, path):

        with open(os.path.join(path, "teacher_success_buffer.pkl"), "wb") as f:
            pickle.dump((self.delta, self.max_size, self.min_stds, self.delta_stds, self.contexts, self.returns,
                         self.delta_reached, self.min_ret, self.get_data()), f)

    def load(self, path):
        with open(os.path.join(path, "teacher_success_buffer.pkl"), "rb") as f:
            self.delta, self.max_size, self.min_stds, self.delta_stds, self.contexts, self.returns, \
            self.delta_reached, self.min_ret, data = pickle.load(f)
        self.set_data(data)


class WassersteinSuccessBuffer(AbstractSuccessBuffer):

    def __init__(self, delta: float, n: int, ep_per_update: int, epsilon: float,
                 context_bounds: Tuple[np.ndarray, np.ndarray], max_reuse=3):
        super().__init__(delta, n, epsilon, context_bounds)
        self.max_reuse = max_reuse
        self.solver = AssignmentSolver(ep_per_update, n, max_reuse=self.max_reuse, verbose=False)
        self.last_assignments = None

    def update_delta_not_reached(self, contexts: np.ndarray, returns: np.ndarray,
                                 current_samples: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, List[bool]]:
        # Only add samples that have a higher return than the median return in the buffer (we do >= here to allow
        # for binary rewards to work)
        med_idx = self.returns.shape[0] // 2
        mask = returns >= self.returns[med_idx]
        n_new = np.sum(mask)
        print("Improving buffer quality with %d samples" % n_new)

        # We do not want to shrink the buffer
        offset_idx = med_idx + 1
        if n_new < offset_idx:
            offset_idx = n_new

        new_returns = np.concatenate((returns[mask], self.returns[offset_idx:]), axis=0)
        new_contexts = np.concatenate((contexts[mask, :], self.contexts[offset_idx:, :]), axis=0)
        sort_idxs = np.argsort(new_returns)

        # Ensure that the buffer is only growing, never shrinking and that all the buffer sizes are consistent
        assert self.contexts.shape[0] <= new_contexts.shape[0]
        assert new_contexts.shape[0] == new_returns.shape[0]

        # These are the indices of the tasks that have NOT been added to the buffer (so the negation of the mas)
        rem_mask = ~mask

        # Ensure that we are not larger than the maximum size
        if new_returns.shape[0] > self.max_size:
            sort_idxs = sort_idxs[-self.max_size:]
            # Since we are clipping potentially removing some of the data chunks we need to update the remainder mask
            # Since we add the new samples at the beginning of the new buffers, we are interested whether the idxs
            # in [0, n_new) are still in the sort_idxs array. If this is NOT the case, then the sample has NOT been
            # added to the buffer.
            rem_mask[mask] = [i not in sort_idxs for i in np.arange(n_new)]

        new_delta_reached = self.returns[self.returns.shape[0] // 2] > self.delta
        return new_delta_reached, new_contexts[sort_idxs, :], new_returns[sort_idxs], rem_mask

    def update_delta_reached(self, contexts: np.ndarray, returns: np.ndarray, current_samples: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, List[bool]]:
        # Compute the new successful samples
        mask = returns >= self.delta
        n_new = np.sum(mask)

        if n_new > 0:
            remove_mask = self.returns < self.delta
            if not np.any(remove_mask) and self.max_reuse * self.returns.shape[0] >= current_samples.shape[0]:
                # At this stage we use the optimizer
                assignments = self.solver(self.contexts, contexts[mask], current_samples, self.last_assignments)
                source_idxs, target_idxs = np.where(assignments)

                # Select the contexts using the solution from the MIP solver. The unique functions sorts the data
                ret_idxs = np.unique(source_idxs)
                new_contexts = np.concatenate((self.contexts, contexts[mask, :]), axis=0)[ret_idxs, :]
                new_returns = np.concatenate((self.returns, returns[mask]), axis=0)[ret_idxs]

                # We update the mask to indicate only the kept samples
                mask[mask] = [idx in (source_idxs - self.contexts.shape[0]) for idx in np.arange(n_new)]

                # We need to relabel the assignments
                up_ret_idxs = np.select([source_idxs == idx for idx in ret_idxs], np.arange(ret_idxs.shape[0]).tolist(),
                                        source_idxs)
                self.last_assignments = (up_ret_idxs, target_idxs)
                avg_dist = np.mean(np.linalg.norm(new_contexts[up_ret_idxs] - current_samples[target_idxs], axis=-1))
                print("Updated success buffer with %d samples. New Wasserstein distance: %.3e" % (n_new, avg_dist))
            else:
                # We replace the unsuccessful samples by the successful ones
                if n_new < np.sum(remove_mask):
                    remove_idxs = np.argpartition(self.returns, kth=n_new)[:n_new]
                    remove_mask = np.zeros(self.returns.shape[0], dtype=bool)
                    remove_mask[remove_idxs] = True

                new_returns = np.concatenate((returns[mask], self.returns[~remove_mask]), axis=0)
                new_contexts = np.concatenate((contexts[mask, :], self.contexts[~remove_mask, :]), axis=0)

                if new_returns.shape[0] > self.max_size:
                    new_returns = new_returns[:self.max_size]
                    new_contexts = new_contexts[:self.max_size, :]

                # Ensure that the buffer is only growing, never shrinking and that all the buffer sizes are consistent
                assert self.contexts.shape[0] <= new_contexts.shape[0]
                assert new_contexts.shape[0] == new_returns.shape[0]
        else:
            new_contexts = self.contexts
            new_returns = self.returns

        return new_contexts, new_returns, ~mask


class AbstractSampler(ABC):

    def __init__(self, context_bounds: Tuple[np.ndarray, np.ndarray]):
        self.noise = 1e-3 * (context_bounds[1] - context_bounds[0])

    def update(self, context: np.ndarray, ret: float) -> NoReturn:
        pass

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        return self.select(samples) + np.random.uniform(-self.noise, self.noise)

    @abstractmethod
    def select(self, samples: np.ndarray) -> np.ndarray:
        pass

    def save(self, path: str) -> NoReturn:
        pass

    def load(self, path: str) -> NoReturn:
        pass


class UniformSampler(AbstractSampler):

    def __init__(self, context_bounds: Tuple[np.ndarray, np.ndarray]):
        super(UniformSampler, self).__init__(context_bounds)

    def select(self, samples: np.ndarray) -> np.ndarray:
        return samples[np.random.randint(0, samples.shape[0]), :]
