# This is the base file that calls everything
import string
import gymnasium
import pygame
from demonstrations.teleop import collect_demos
import torch
import subprocess
from pyfiglet import Figlet
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="teleop")



def print_fancy_text(text):
    f = Figlet(font='small')  # You can choose different fonts
    print(f.renderText(text))


if __name__ == "__main__":

    print_fancy_text("HUMAN AI ALIGNMENT PROJECT")

    # Prompt the user to enter the type of feedback
    feedback_type = input("Enter the type of feedback (Demonstrations: D, Comparisons: C, off: O, improvement: I): ").upper()

    # Choose the appropriate feedback function based on the provided feedback type

    # Demonstrations
    if feedback_type == "D":
        # Code to call demonstrations and BCO
        subprocess.call(["python", "demonstrations/mountain_car_bco.py"])

    # Comparisons
    elif feedback_type == "C":
        # Call the comparison python script
        pass
        

    # Off
    elif feedback_type == "O":
        pass

    # Improvement
    elif feedback_type == "I":
        pass

    else:
        print("Invalid feedback type. Please choose from (Demonstrations: D, Comparisons: C, off: O, improvement: I)")
