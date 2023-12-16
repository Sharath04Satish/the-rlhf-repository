# This is the base file that calls everything
import subprocess
from pyfiglet import Figlet
import warnings
import os

warnings.filterwarnings("ignore", category=DeprecationWarning, module="teleop")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


def print_fancy_text(text):
    f = Figlet(font="small")  # You can choose different fonts
    print(f.renderText(text))


if __name__ == "__main__":
    print_fancy_text("HUMAN AI ALIGNMENT PROJECT")

    print("Ankith Boggaram, Sharath Satish\n")

    print("Before proceeding, please make note of the following:\n")

    print(
        "For demonstrations, please use the arrow keys to control the car and proceed to demonstrate the task 3 times"
    )
    print(
        "For comparisons, please take a look at the two shown trajectories and choose the one that you think is better"
    )
    print(
        "For improvement, please take a look at the shown trajectory and try to improve it by demonstrating a better trajectory"
    )

    # Prompt the user to enter the type of feedback
    feedback_type = input(
        "Enter the type of feedback (Demonstrations: D, View Comparisons: C, Run Comparisons: A, View Improvement: I, Run Improvement: B): "
    ).upper()

    # Choose the appropriate feedback function based on the provided feedback type

    # Demonstrations
    if feedback_type == "D":
        # Code to call demonstrations and BCO
        subprocess.call(["python3", "demonstrations/cartpole_bc.py"])

    # Comparisons
    elif feedback_type == "C":
        # Call the comparison python3 script
        os.chdir("web_app")
        subprocess.call(["python3", "-m", "flask", "run"])

    elif feedback_type == "A":
        # Call the comparison python3 script
        subprocess.call(["python3", "comparisons/integrations.py"])

    elif feedback_type == "I":
        # Call the comparison python3 script
        os.chdir("web_app")
        subprocess.call(["python3", "-m", "flask", "run"])

    # Improvement
    elif feedback_type == "B":
        subprocess.call(["python3", "improvement/improvement_integrations.py"])

    else:
        print(
            "Invalid feedback type. Please choose from (Demonstrations: D, Comparisons: C, improvement: I)"
        )
