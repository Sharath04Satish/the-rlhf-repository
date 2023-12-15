import subprocess
from offline_reward_learning import generate_training_data, learn_reward_function

value = generate_training_data()

if value != False:
    (
        reward_net,
        optimizer,
        training_pairs,
        training_labels,
        num_iter,
        checkpoint,
    ) = value

    learn_reward_function(
        reward_net, optimizer, training_pairs, training_labels, num_iter, checkpoint
    )

    subprocess.call(
        [
            "python3",
            "improvement/vpg.py",
            "--epochs",
            "40",
            "--checkpoint",
            "--reward",
            "improvement/reward.params",
            "--checkpoint_dir",
            "improvement/rlhf",
        ]
    )
    subprocess.call(
        [
            "python3",
            "improvement/rollout_policy.py",
            "--checkpoint",
            "improvement/rlhf/policy_checkpoint39.params",
            "--num_rollouts",
            "5",
            "--render",
        ]
    )
