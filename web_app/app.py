from flask import Flask, render_template, request
from constants import num_synthetic_demonstrations
from random import choice, randint
import json
import os
import signal

app = Flask(__name__)

counter = 1
preferences = list()
improvement_indices = list()


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


def get_single_comparison_videos():
    comparison_videos = list()

    for _ in range(5):
        demonstration_a_index = randint(1, 10)

        demonstration_a = (
            "./static/comparisons_data/synthetic_demonstration_{0}.mp4".format(
                demonstration_a_index
            )
        )

        comparison_videos.append(("instance", demonstration_a_index, demonstration_a))

    for _, index, _ in comparison_videos:
        improvement_indices.append(index)

    with open("../improvement/data/improvement_indices.json", "w") as data_file:
        print("L1", improvement_indices)
        json_data = {"improvement_indices": improvement_indices}
        json.dump(json_data, data_file)

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


@app.route("/improvement", methods=["POST"])
def handle_improvement_inputs():
    global improvement_indices
    print("L1")
    with open("../improvement/data/improvement_indices.json", "w") as data_file:
        print("L1", improvement_indices)
        json_data = {"improvement_indices": improvement_indices}
        json.dump(json_data, data_file)

    os.kill(os.getpid(), signal.SIGINT)

    return "Thank you for your input. Please run integrations.py to create the training data and learn a reward function."


@app.route("/improvement")
def render_random_trajectories_imprv():
    comparison_videos = get_single_comparison_videos()
    print(comparison_videos)
    return render_template("improvements.html", comparison_videos=comparison_videos)


@app.route("/shutdown")
def shutdown():
    os.kill(os.getpid(), signal.SIGINT)
    return "Server shutting down..."


def shutdown_server():
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    func()
