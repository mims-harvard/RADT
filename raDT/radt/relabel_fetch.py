"""
Hindsight avoid-region relabeling for FetchReachObstacle (Appendix A.3).

Pass 1: one avoid box per trajectory, centered in a concave "nook" of the trajectory's
contour (contour-based sampling) or, if there is none, on the segment between the start
and end positions. z = 1 if the trajectory never enters the box.
Pass 2: a copy of every trajectory, with boxes resampled on the start-end segment until
z is the opposite of pass 1.
"""
import argparse
import copy
import numpy as np
from concave_hull import convex_hull_indexes, concave_hull_indexes

from relabel_common import load_random_policy_data, avoids_all, save_dataset

STATE_DIM = 10  # FetchReach observation without the obstacle features


def contour_centroids(traj):
    # candidate avoid centroids: midpoints of the ends of concave nooks in the trajectory contour
    positions = traj['observations'][:, :3]
    concave_idx = concave_hull_indexes(positions, concavity=1)
    convex_idx = convex_hull_indexes(positions)
    in_nook = ~np.isin(concave_idx, convex_idx)

    # group consecutive concave-hull points that are not on the convex hull
    groups, group_id, prev = [], 0, False
    for b in in_nook:
        if not b:
            groups.append(0)
            prev = False
        else:
            if not prev:
                group_id += 1
            groups.append(group_id)
            prev = True
    groups = np.array(groups)
    if in_nook[0] and in_nook[-1]:
        groups[groups == group_id] = 1

    # each nook, including its two bordering convex-hull points; keep nooks with >= 10 points
    n = len(groups)
    nooks = []
    for group in range(1, max(groups) + 1):
        idxs = []
        for i in range(n):
            if group in (groups[(i - 1) % n], groups[i % n], groups[(i + 1) % n]) and concave_idx[i] not in idxs:
                idxs.append(concave_idx[i])
        if len(idxs) >= 10:
            nooks.append(idxs)

    # keep centroids roughly between the start and end of the trajectory
    centroids = []
    start, end = positions[0], positions[-1]
    for idxs in nooks:
        c = positions[[idxs[0], idxs[-1]], :].mean(axis=0)
        v1, v2 = end - start, c - start
        v3, v4 = start - end, c - end
        angle1 = np.arccos(np.clip(np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)), -1.0, 1.0))
        angle2 = np.arccos(np.clip(np.dot(v3 / np.linalg.norm(v3), v4 / np.linalg.norm(v4)), -1.0, 1.0))
        if abs(angle1) <= np.pi / 6 and abs(angle2) <= np.pi / 6:
            centroids.append(c)
    return np.array(centroids)


def box(center, half_width):
    return np.concatenate([center - half_width, center + half_width])


def relabel_pass1(traj, max_half_width):
    positions = traj['observations'][:, :3]
    start, end = positions[0], positions[-1]
    centroids = contour_centroids(traj)
    if not centroids.shape[0]:
        center = start + (end - start) * np.random.rand()
    else:
        center = centroids[0]
    boxes = np.array([box(center, (0.1 + 0.9 * np.random.rand()) * max_half_width)])
    traj['avoid_states'] = boxes
    traj['success'] = avoids_all(positions, boxes)
    return traj['success']


def relabel_pass2(traj, paired_success, max_half_width):
    positions = traj['observations'][:, :3]
    start, end = positions[0], positions[-1]
    success = paired_success
    while success == paired_success:
        center = start + (end - start) * np.random.rand()
        boxes = np.array([box(center, np.random.rand() * max_half_width)])
        success = avoids_all(positions, boxes)
    traj['avoid_states'] = boxes
    traj['success'] = success
    return success


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_half_width', type=float, default=0.08, help='maximum half-width of relabeled avoid boxes')
    parser.add_argument('--output', type=str, default='reach_obstacle_relabeled.pkl', help='file name in RADT_DATASET_DIR')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    trajectories = load_random_policy_data('FetchReachObstacle', 'buffer_massless')
    copies = copy.deepcopy(trajectories)

    successes1, successes2 = [], []
    for traj, traj2 in zip(trajectories, copies):
        success1 = relabel_pass1(traj, args.max_half_width)
        successes1.append(success1)
        successes2.append(relabel_pass2(traj2, success1, args.max_half_width))
        traj['observations'] = traj['observations'][:, :STATE_DIM]
        traj2['observations'] = traj2['observations'][:, :STATE_DIM]

    print('fraction of successful avoid demonstrations: pass 1 {:.3f}, pass 2 {:.3f}'.format(np.mean(successes1), np.mean(successes2)))
    save_dataset(trajectories + copies, args.output)
