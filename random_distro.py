#!/usr/bin/python3.8
"""
Algorithm to compare Genetic Algorithm to Random Sample Results
"""

# Needed functions
import numpy as np
import matplotlib.pyplot as plt
from panopticam import gen_randpop, rate_gen, fitness_function
from test_pop import test_pop
from typing import Dict, List
# Needed constants
from panopticam import (
    view_radius,
    genetic_pop,
    cam_count,
    citizens,
    test_number,
)

threshold = 120
generation = 1000


def plot_final_graph(generational_record: Dict[int, int], ) -> None:
    plt.cla()
    plt.gcf().canvas.mpl_connect(
        "key_release_event", lambda event: [exit(0) if event.key == "escape" else None]
    )
    plt.grid(True)
    lists = sorted(generational_record.items())
    x, y = zip(*lists)
    avg = [sum(y) / len(y)] * len(x)
    mean = np.mean(y)
    std_dev = format(np.std(y), '.3f')
    plt.scatter(x, y, label="Pop Surveilled", color="r")
    ax = plt.gca()
    ax.set_xlabel("Generations")
    ax.set_ylabel("Population Surveilled")
    maximum = generational_record[max(generational_record, key=generational_record.get)]
    ax.set_title(f"Population Surveilled Over Generations using Random Generation\nPopulation Surveilled Mean {mean}, Max {maximum}, Stddev {std_dev}")
    ax.plot(x, avg, label='Mean', linestyle='--')
    ax.legend(loc='upper right')
    plt.savefig(f'./results/final_random_test_gen_{generation}.png')
    plt.show()


if __name__ == "__main__":

    generational_record = {}  # record to graph at end
    cams = {}  # dictionary of cam population
    #citpop = gen_randpop(citizens)  # a numpy array of citizens randomly distributed
    citpop = np.array(test_pop)
    # Main Algorithm Loop
    gen = 0
    max_seen = 0
    while (
        (gen < generation) & (max_seen < threshold)
    ): # evolve for number of generations or until threshold

        for i in range(genetic_pop):  # generate population
            cams[i] = gen_randpop(
                cam_count
            )  # a numpy array of cams randomly distributed

        total_scores = rate_gen(cams, citpop)  # get scores of cams
        best_cam = max(total_scores, key=total_scores.get)  # get best score
        max_seen = total_scores[best_cam]  # get number of best score

        # print(
        #     f"We surveilled {max_seen} in generation {gen}, best is {best_cam}"
        # )  # print results

        generational_record[gen] = max_seen  # to graph at end of process
        gen += 1

    plot_final_graph(generational_record) # produces and saves final plot
