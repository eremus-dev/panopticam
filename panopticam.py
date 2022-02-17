#!/usr/bin/python3.8
"""
Genetic Algorithm to maximize surveillance over a population for AI Assignment.

Author: Sam (eremus-dev)
Repo: https://github.com/eremus-dev
"""

import math
from collections import Counter
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from test_pop import test_pop

""" 
GENETIC ALGORITHM CONFIGURATION VARIABLES
"""
# Genetic Algorithm and Camera Config
genetic_pop = 100  # number different genetic strains
generation = 100  # number of generations to maximize coverage
view_radius = 15  # how far the cameras see
citizens = 200  # how many people we need to surveil
cam_count = 4  # how many cams we have to surveil them with
mutation_chance = 10  # percentage chance mutation occurs
threshold = 100  # stop at this result or generation
test_number = 10 # number of tests to run, set to zero if no tests

Coord = List[int]  # Type of co-ordinates


def gen_randpop(size: int) -> List[Coord]:
    """
    Function to generate randomly distributed population
    to surveil
    """
    obs = []  # [x,y] of size number of people
    for _ in range(1, size + 1):
        xy = []  # x, y co-ords of each person

        x = np.random.randint(1, 100)
        xy.append(x)

        y = np.random.randint(1, 100)
        xy.append(y)
        obs.append(xy)

    return np.array(obs, copy=True)


def rate_gen(cams: Dict[int, List[Coord]], pop: List[Coord]) -> Dict[int, int]:
    """
    Function to get the best of the population to breed, mutate and to survive
    """
    scores = {}
    for n in cams:
        scores[n] = fitness_function(cams[n], pop)

    return scores


def fitness_function(cams: List[Coord], pop: List[Coord]) -> int:
    """
    Function to calculate number of surveilled citizens.
    Check if all the cameras can see them, if any can score increases
    """
    score = []

    for cit in pop:
        test = False
        for cam in cams:
            if (
                math.sqrt(((cam[0] - cit[0]) ** 2) + ((cam[1] - cit[1]) ** 2))
                <= view_radius
            ):
                test = True
        score.append(test)
    return score.count(True)


def select_from_pop(
    cams: Dict[int, List[Coord]], total_scores
) -> Dict[int, List[Coord]]:
    """
    Function that takes a dict of camera positions and a dict of scores and breeds the strongest
    returns new population of cameras
    """
    top_scores = {}
    new_pop = {}
    selection = int(len(total_scores) / 2)
    scores = sorted(total_scores, key=total_scores.get, reverse=True)[:selection]

    assert len(scores) == selection

    for i in scores:
        top_scores[i] = total_scores[i]
        new_pop[i] = cams[i]

    assert len(new_pop) == selection
    return breed_strongest(top_scores, new_pop)


def breed_strongest(
    top_scores: Dict[int, int], new_pop: Dict[int, List[Coord]]
) -> Dict[int, List[Coord]]:
    """
    Function to breed 25 best positions.
    Strongest always remains unchanged.
    """
    count = 0
    full_pop = {}
    keys = list(new_pop.keys())

    for i in keys:

        dad = []
        child = []
        mum = []

        mum = np.copy(new_pop[i])
        child = dad = np.copy(
            new_pop[np.random.choice(keys)]
        )  # randomly select breeding mate

        child[0] = mum[np.random.randint(0, 3)]
        child[1] = mum[np.random.randint(0, 3)]

        full_pop[count] = mum  # save mum
        count += 1
        full_pop[count] = child  # add random child
        count += 1

    full_pop = mutate(full_pop, top_scores)
    assert len(full_pop) == genetic_pop
    return full_pop


def mutate(
    full_pop: Dict[int, List[Coord]], top_scores: Dict[int, int]
) -> Dict[int, List[Coord]]:
    """
    Function to mutate population, 10% chance they will mutate
    """
    for i in full_pop:
        if np.random.randint(0, 100) > (100 - mutation_chance):
            temp = full_pop[i]
            xmod, ymod = [
                np.random.randint(-20, 20),
                np.random.randint(-20, 20),
            ]  # pick random mutation
            camera_num = np.random.randint(0, 3)
            camera = temp[camera_num] # cameras to mod
            camera[0] =  (camera[0] + xmod) % 100
            camera[1] = (camera[1] + ymod) % 100
            temp[camera_num] = camera
            full_pop[i] = temp

    return full_pop


def plot_pop(pop: List[Coord], cams: List[Coord], top_score: int, gen: int, run: int) -> None:
    """
    Function to plot placement of cams and population on graph
    """
    plt.cla()                              # clears graph
    plt.gcf().canvas.mpl_connect(          # allows exit key to quit qraph
        "key_release_event", lambda event: [exit(0) if event.key == "escape" else None]
    )
    plt.axis("equal")                       
    plt.grid(True)
    plt.plot(pop[:, 0], pop[:, 1], "ok")
    plt.plot(cams[:, 0], cams[:, 1], "*")

    for i in range(len(cams)):      # plots camera view range
        circle = plt.Circle(
            (cams[i][0], cams[i][1]), view_radius, color="r", fill=False
        )
        ax = plt.gca()
        ax.add_artist(circle)
    ax = plt.gca()
    ax.set_xlabel("City Terrain X")     # sets up all labels
    ax.set_ylabel("City Terrain Y")
    ax.set_title(f"Visualisation of Cameras and Population\nSurveilled Population {max_seen} in Generation {gen}")
    plt.pause(0.01)
    plt.draw()                          # draws graph

    if gen == 199:
        plt.savefig(f'./results/last_gen_test{run}.png')


def plot_final_results(generational_record: Dict[int, int], run: int, max_seen: int) -> None:
    '''
    Produces final plot of the progression of the GA across a single generational run
    '''
    plt.cla()
    plt.grid(True)
    lists = sorted(generational_record.items())
    x, y = zip(*lists)
    plt.xlim(-2, generation+2)
    plt.ylim(50, 120)
    plt.plot(x, y, label="Pop Surveilled", linestyle="--", marker='o')
    ax = plt.gca()
    ax.set_xlabel("Generations")
    ax.set_ylabel("Number of Population Surveilled")
    ax.set_title(f"Population Surveilled Over Generations\nMax Population Surveilled {max_seen}")
    plt.savefig(f'./results/final_results_test{run}.png')
    
    if test_number > 0:
        plt.pause(0.5)
        plt.draw()
    else:
        plt.show()


def plot_aggregate_results(aggregate_results: Dict[int, int], ) -> None:
    '''
    Produces plot of aggregate results for test runs of GA
    '''
        # Graph aggregate results and average of test runs
    plt.cla()
    plt.grid(True)
    lists = sorted(aggregate_results.items())
    x,y = zip(*lists)
    avg = [sum(y) / len(y)] * len(x)
    mean = np.mean(y)
    std_dev = format(np.std(y), '.3f')
    maximum = max(y)
    plt.scatter(x, y, label="Pop Surveilled", color="r")
    ax = plt.gca()
    ax.plot(x, avg, label='Mean', linestyle='--')
    ax.set_title(f"Population Surveilled Over Tests using Genetic Algorithm\nPopulation Surveilled Mean: {mean}, Max {maximum}, Stdev {std_dev}")
    ax.legend(loc='upper left')
    ax.set_xlabel("Test Number")
    ax.set_ylabel("Number of Population Surveilled")
    plt.savefig(f'./results/aggregate_result_GA_test_run.png')
    plt.show()



if __name__ == "__main__":
    
    aggregate_results = {} # collect each tests results
    # run the GA for test_number times and graph results
    for run in range(0, test_number): 
        generational_record = {}  # record to graph at end
        cams = {}  # dictionary of genetic population
        #citpop = gen_randpop(citizens)  # a numpy array of citizens randomly distributed
        citpop = np.array(test_pop)

        for i in range(genetic_pop):  # generate genetic population
            cams[i] = gen_randpop(cam_count)  # a numpy array of cams randomly distributed

        # Main Genetic Algorithm Loop
        gen = 0
        max_seen = 0
        while (gen < generation) & (
            max_seen < threshold
        ):  # evolve for number of generations

            if gen != 0:  # do nothing first time through loop
                cams = select_from_pop(cams, total_scores)

            total_scores = rate_gen(cams, citpop)

            best_cam = max(total_scores, key=total_scores.get)
            max_seen = total_scores[best_cam]

            print(f"We surveilled {max_seen} in generation {gen}, best is {best_cam}")
            plot_pop(citpop, cams[best_cam], max_seen, gen, run)  # print best fit for each generation

            generational_record[gen] = max_seen  # to graph at end of process
            gen += 1

        # Graph Results of Genetic Algorithm over generations
        plot_final_results(generational_record, run, max_seen)

        aggregate_results[run] = max_seen

    # Graph aggregate results and average of test runs
    plot_aggregate_results(aggregate_results)
    
