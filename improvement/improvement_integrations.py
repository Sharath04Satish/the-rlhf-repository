from offline_reward_learning import get_store_novice_demonstrations
from offline_reward_learning import generate_training_data
from offline_reward_learning import learn_reward_function

while True:
    try:
        option = int(
            input(
                "Select an option\nEnter 1 for creating training data and learn reward function \n"
            )
        )
        break
    except:
        print("Let's try that one more time.")

if option == 1:
    value = generate_training_data()

    if value != False:
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
