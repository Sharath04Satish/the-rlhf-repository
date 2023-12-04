from offline_reward_learning import get_store_novice_demonstrations
from offline_reward_learning import generate_training_data
from offline_reward_learning import learn_reward_function

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
else:
    print("Let's try that one more time!")
