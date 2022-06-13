import functools
import warnings

from collections import deque
from sklearn.cluster import KMeans
import numpy as np
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class State_Clustering:
    def __init__(self, n_cluster, name=''):
        self.name = name
        self.n_cluster = n_cluster
        self.cluster_idx_to_id = None
        self.max_distance = [-np.inf for _ in range(n_cluster)]
        self.sample_buff = [deque(maxlen=1000) for _ in range(n_cluster)]
        self.k_means = KMeans(n_clusters=n_cluster)
        self.is_initialized = False

    def n_active_clusters(self):
        if not self.is_initialized:
            return 0
        else:
            return len(self.cluster_idx_to_id)

    def _relabel(self):
        """
        relabel each cluster based on minimum label changes
        """
        def mapping_cmp(a, b):
            # check if a or b is in current mapping
            if a[1] < b[1]:
                return 1
            elif a[1] > b[1]:
                return -1
            else:
                a_cluster_idx, a_cluster_id = a[0]
                b_cluster_idx, b_cluster_id = b[0]
                if a_cluster_idx in self.cluster_idx_to_id and a_cluster_id == self.cluster_idx_to_id[a_cluster_idx]:
                    return -1
                elif b_cluster_idx in self.cluster_idx_to_id and b_cluster_id == self.cluster_idx_to_id[b_cluster_idx]:
                    return 1
                else:
                    return 0

        cluster_mapping_candidates = dict()
        all_samples = []
        curr_cluster_id = []
        for cluster_id in range(len(self.sample_buff)):
            if len(self.sample_buff[cluster_id]) > 0:
                for sample in self.sample_buff[cluster_id]:
                    curr_cluster_id.append(cluster_id)
                    all_samples.append(sample)
        # make prediction
        pred_cluster = self.k_means.predict(all_samples)
        for sample_idx, pred_c in enumerate(pred_cluster):
            k = (pred_c, curr_cluster_id[sample_idx])
            if k not in cluster_mapping_candidates:
                cluster_mapping_candidates[k] = 1
            else:
                cluster_mapping_candidates[k] += 1
        # assign labels
        new_cluster_idx_to_id = dict()
        is_id_assigned = [False for _ in range(self.n_cluster)]
        sorted_mapping = sorted(cluster_mapping_candidates.items(), key=functools.cmp_to_key(mapping_cmp))
        for m in sorted_mapping:
            cluster_idx, cluster_id = m[0]
            if cluster_idx in new_cluster_idx_to_id:
                continue
            else:
                if not is_id_assigned[cluster_id]:
                    new_cluster_idx_to_id[cluster_idx] = cluster_id
                    is_id_assigned[cluster_id] = True
                else:
                    # get an assignable id
                    for i in range(self.n_cluster):
                        if not is_id_assigned[i]:
                            is_id_assigned[i] = True
                            new_cluster_idx_to_id[cluster_idx] = i
                            break

        new_sample_buff = [deque(maxlen=1000) for _ in range(self.n_cluster)]
        for i_sample, sample in enumerate(all_samples):
            new_sample_buff[new_cluster_idx_to_id[pred_cluster[i_sample]]].append(sample)
        self.sample_buff = new_sample_buff
        self.cluster_idx_to_id = new_cluster_idx_to_id

        # re-calculate the centroid distance
        self.max_distance = [-np.inf for _ in range(self.n_cluster)]
        centroid_distances = self.k_means.transform(all_samples)
        for i in range(len(all_samples)):
            cluster_id = new_cluster_idx_to_id[pred_cluster[i]]
            self.max_distance[cluster_id] = max(self.max_distance[cluster_id], centroid_distances[i, pred_cluster[i]])

        print(f'[INFO:Cluster:{self.name}] update clustering, n active clusters: {self.n_active_clusters()}')

    def extend_states(self, states):
        for s in states:
            self.add_state(s)

    def add_state(self, state):
        if not self.is_initialized:
            self.k_means.fit([np.copy(state) for _ in range(self.n_cluster)])
            self.sample_buff[0].extend([np.copy(state) for _ in range(self.n_cluster)])
            self.is_initialized = True
            self.max_distance[0] = 0
            self.cluster_idx_to_id = {0: 0}
            self.update_clustering()
        else:
            pred_cluster_idx = self.k_means.predict([state])[0]
            dist_to_centroid = self.k_means.transform([state])[0, pred_cluster_idx]
            assert pred_cluster_idx in self.cluster_idx_to_id, f'{pred_cluster_idx} not in {self.cluster_idx_to_id}'
            cluster_id = self.cluster_idx_to_id[pred_cluster_idx]
            self.sample_buff[cluster_id].append(state)
            # check if clustering update is needed
            if dist_to_centroid > self.max_distance[cluster_id] + 1e-5:
                self.update_clustering()

    def get_cluster_id(self, state):
        assert self.is_initialized
        return self.cluster_idx_to_id[self.k_means.predict([state])[0]]

    def get_clusters(self, states):
        assert self.is_initialized
        return [self.cluster_idx_to_id[p] for p in self.k_means.predict(states)]

    def update_clustering(self):
        all_samples = []
        for cluster_id in range(len(self.sample_buff)):
            if len(self.sample_buff[cluster_id]) > 0:
                all_samples.extend(self.sample_buff[cluster_id])
        self.k_means.fit(all_samples)
        self._relabel()


def main():
    """ For debugging """
    s_cluster = State_Clustering(n_cluster=4)
    samples = [[1., 1.], [1., 1.], [1., 1.]]
    s_cluster.extend_states(samples)
    print(s_cluster.get_clusters(samples))

    s_cluster.add_state([2., 2.])
    samples.append([2., 2.])
    print(s_cluster.get_clusters(samples))

    s_cluster.add_state([3., 3.])
    samples.append([3., 3.])
    print(s_cluster.get_clusters(samples))

    s_cluster.add_state([1., 1.])
    samples.append([1., 1.])
    print(s_cluster.get_clusters(samples))

    s_cluster.add_state([4., 4.])
    samples.append([4., 4.])
    print(s_cluster.get_clusters(samples))


if __name__ == '__main__':
    main()









