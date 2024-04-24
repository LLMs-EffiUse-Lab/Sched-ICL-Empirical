import random

import matplotlib.pyplot as plt
from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.core.problem import IntegerProblem, IntegerSolution
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.lab.visualization.plotting import Plot
from jmetal.util.solution import get_non_dominated_solutions
import time
import pandas as pd
import os
import numpy as np


class AllocationProblem(IntegerProblem):
    def __init__(self, df_pre_accuracy, df_true_accuracy, df_cost, model_list):
        super(AllocationProblem, self).__init__()
        self.df_pre_accuracy = df_pre_accuracy
        self.df_true_accuracy = df_true_accuracy
        self.df_cost = df_cost
        self.model_list = model_list

        self.number_of_variables = len(df_true_accuracy)
        self.obj_directions = [self.MINIMIZE, self.MAXIMIZE]
        self.obj_labels = ['Cost', 'Expected Accuracy']

        self.lower_bound = [0] * self.number_of_variables
        self.upper_bound = [len(model_list) - 1] * self.number_of_variables

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        s_cost = np.zeros(len(solution.variables))
        s_accuracy = np.zeros(len(solution.variables))
        s_allocation = solution.variables.copy()
        for i, allocation in enumerate(s_allocation):
            s_cost[i] = self.df_cost.iloc[i, allocation]
            s_accuracy[i] = self.df_pre_accuracy.iloc[i, allocation]
        s_total_cost = np.sum(s_cost)
        s_accuracy_mean = np.mean(s_accuracy)
        solution.objectives[0] = s_total_cost
        solution.objectives[1] = -s_accuracy_mean
        # cost = 0
        # expected_accuracy = 0
        # for i in range(len(solution.variables)):
        #     if solution.variables[i] < 0 or solution.variables[i] >= len(self.model_list):
        #         raise Exception(f'Invalid model index: {solution.variables[i]}')
        #     index = solution.variables[i]
        #     model_column = self.model_list[index]
        #     cost += self.df_cost[model_column][solution.variables[index]]
        #     expected_accuracy += self.df_pre_accuracy[model_column][solution.variables[index]]
        # solution.objectives[0] = cost
        # solution.objectives[1] = -expected_accuracy  # Negative value for maximization
        return solution

    def create_solution(self) -> IntegerSolution:
        new_solution = IntegerSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives(),
            self.number_of_constraints()
        )
        new_solution.variables = [
            random.randint(self.lower_bound[i], self.upper_bound[i])
            for i in range(self.number_of_variables)
        ]
        return new_solution

    def name(self) -> str:
        return 'Allocation Problem'

    def number_of_objectives(self) -> int:
        return 2

    def number_of_constraints(self) -> int:
        return 0




class spea2(object):
    def __init__(self, df_pre_accuracy, df_true_accuracy, df_cost, model_list, termination):
        self.df_pre_accuracy = df_pre_accuracy
        self.df_true_accuracy = df_true_accuracy
        self.df_cost = df_cost
        self.model_list = model_list
        self.termination = termination

    def eval_solutions(self, solution):
        cost = 0
        expected_accuracy = 0
        true_accuracy = 0
        for i in range(len(solution)):
            cost += self.df_cost.iloc[i, solution[i]]
            expected_accuracy += self.df_pre_accuracy.iloc[i, solution[i]]
            true_accuracy += self.df_true_accuracy.iloc[i, solution[i]]
        exp_acc = expected_accuracy / len(solution)
        true_acc = true_accuracy / len(solution)
        return cost, exp_acc, true_acc

    def run(self):
        start_time = time.time()
        problem = AllocationProblem(self.df_pre_accuracy, self.df_true_accuracy, self.df_cost, self.model_list)
        algorithm = SPEA2(
            problem=problem,
            population_size=100,
            offspring_population_size=100,
            mutation=PolynomialMutation(probability=0.2, distribution_index=20),
            crossover=SBXCrossover(probability=1.0, distribution_index=20),
            termination_criterion=StoppingByEvaluations(max_evaluations=self.termination)
        )
        print("Start SPEA2 searching!")
        algorithm.run()
        elapsed_time = time.time() - start_time

        solutions = algorithm.get_result()
        spea2_res = []
        for i in range(len(solutions)):
            cost, exp_acc, true_acc = self.eval_solutions(solutions[i].variables)
            spea2_res.append([cost, exp_acc, true_acc])
        spea2_res = pd.DataFrame(spea2_res, columns=['cost', 'expected_accuracy', 'true_accuracy'])
        spea2_res['time'] = elapsed_time

        fig = plt.figure(figsize=(8, 5))
        plt.scatter(spea2_res['cost'], spea2_res['true_accuracy'], color='blue')
        plt.xlabel('$f_{cost}}$ (USD)')
        plt.ylabel(r"$f_{acc}}$ ")
        plt.show()
        print("SPEA2 finished and the searchiing time is: ", elapsed_time)
        return spea2_res







