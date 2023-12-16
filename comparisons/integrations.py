from offline_reward_learning_syn import get_store_novice_demonstrations
from offline_reward_learning_syn import generate_training_data
from offline_reward_learning_syn import learn_reward_function
import subprocess

while True:
    try:
        option = int(
            input(
                "Select an option\nEnter 1 for generating synthetic demonstrations\nEnter 2 for creating training data and learn reward function \n"
            )
        )
        break
    except:
        print("Let's try that one more time.")

if option == 1:
    get_store_novice_demonstrations()
elif option == 2:
    (
        reward_net,
        optimizer,
        training_pairs,
        training_labels,
        num_iter,
        checkpoint,
    ) = generate_training_data()
    learn_reward_function(
        reward_net, optimizer, training_pairs, training_labels, num_iter, checkpoint
    )

    subprocess.call(
        [
            "python3",
            "comparisons/vpg.py",
            "--epochs",
            "40",
            "--checkpoint",
            "--reward",
            "comparisons/reward.params",
            "--checkpoint_dir",
            "comparisons/rlhf",
        ]
    )
    subprocess.call(
        [
            "python3",
            "comparisons/rollout_policy_syn.py",
            "--checkpoint",
            "comparisons/rlhf/policy_checkpoint39.params",
            "--num_rollouts",
            "5",
            "--render",
        ]
    )
else:
    print("Let's try that one more time!")
