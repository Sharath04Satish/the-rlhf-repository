from flask import Flask, render_template, request
from constants import num_synthetic_demonstrations
from random import choice, randint
import json

app = Flask(__name__)

counter = 1
# txt_file = open("../comparisons/data/comparisons_preferences.txt", "w")
preferences = list()


def generate_unique_random_numbers():
    random_number1 = randint(1, num_synthetic_demonstrations)
    random_number2 = randint(1, num_synthetic_demonstrations)

    while random_number1 == random_number2:
        random_number2 = randint(1, num_synthetic_demonstrations)

    return random_number1, random_number2


def get_comparison_videos():
    comparison_videos = list()

    for _ in range(num_synthetic_demonstrations):
        demonstration_a_index, demonstration_b_index = generate_unique_random_numbers()

        demonstration_a = (
            "./static/comparisons_data/synthetic_demonstration_{0}.mp4".format(
                demonstration_a_index
            )
        )
        demonstration_b = (
            "./static/comparisons_data/synthetic_demonstration_{0}.mp4".format(
                demonstration_b_index
            )
        )

        comparison_videos.append(
            (
                "instance",
                demonstration_a_index,
                demonstration_a,
                demonstration_b_index,
                demonstration_b,
            )
        )

        return comparison_videos


@app.route("/")
def render_comparisons():
    comparison_videos = get_comparison_videos()

    return render_template(
        "index.html",
        comparison_videos=comparison_videos,
    )


@app.route("/", methods=["POST"])
def handle_comparison_inputs():
    global counter
    global txt_file
    global preferences

    if counter <= 10:
        counter += 1
        if request.method == "POST":
            button_value = request.form.get("instance")
            print("Clicked button:", tuple(button_value))

            preferences.append(button_value)

            return render_template(
                "index.html",
                comparison_videos=get_comparison_videos(),
            )
    else:
        with open("../comparisons/data/comparisons_preferences.json", "w") as data_file:
            json_data = {"preferences": preferences}
            json.dump(json_data, data_file)

        return "Thank you for your inputs. Please run integrations.py to create the training data and learn a reward function."