from abc import ABC, abstractmethod
from logs import *
import numpy as np 
import pandas as pd
import csv
from scipy.stats import norm
import matplotlib.pyplot as plt
import os 
np.random.seed(1)


#--------------------------------------#


# Set up the basic configuration for logging
logging.basicConfig

# Get or create a logger with the specified name
logger = logging.getLogger("MAB Application")

# Create a console handler with a higher log level (INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Set a custom formatter for the console handler
ch.setFormatter(CustomFormatter())

# Add the console handler to the logger
logger.addHandler(ch)

# Set the logger level to INFO to capture INFO logs as well
logger.setLevel(logging.INFO)


#--------------------------------------#


class Bandit(ABC):
##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # log average reward (use f strings to make it informative)
        # log average regret (use f strings to make it informative)
        pass


#--------------------------------------#


class Visualization():
    def __init__(self, algo, bandit_rewards, num_trials):
        self.algo = algo
        self.bandit_rewards = bandit_rewards
        self.num_trials = num_trials

    def plot1(self):
        """ Visualize the performance (convergence, learning process) 
            of each bandit: linear and log """

        # Identify the algorithm
        if self.algo == 'EpsilonGreedy':
            my_algo = EpsilonGreedy(self.bandit_rewards, self.num_trials)
        elif self.algo == 'ThompsonSampling':
            my_algo = ThompsonSampling(self.bandit_rewards, self.num_trials)
        else:
            logger.warning('No such algorithm implemented')
            return False
        
        # Run the experiment and get the cumulative average rewards
        cumulative_rewards = my_algo.experiment()[0]
        
        # Plotting the performance
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Linear scale plot
        axes[0].plot(cumulative_rewards)
        axes[0].set_xlabel("Number of Trials")
        axes[0].set_ylabel("Cumulative Average Reward")
        axes[0].set_title("Convergence of Rewards (Linear Scale)")

        # Add a horizontal line for the maximum reward        
        axes[0].axhline(y=max(self.bandit_rewards), color='r', linestyle='--', label=f'Max Reward: {max(self.bandit_rewards)}')
        axes[0].legend()
        
        # Log scale plot
        axes[1].plot(cumulative_rewards)
        axes[1].set_xlabel("Number of Trials")
        axes[1].set_ylabel("Cumulative Average Reward")
        axes[1].set_title("Convergence of Rewards (Log Scale)")
        axes[1].set_xscale('log')
        
        # Plot the maximum reward
        axes[1].axhline(y=max(self.bandit_rewards), color='r', linestyle='--', label=f'Max Reward: {max(self.bandit_rewards)}')
        axes[1].legend()

        # Add a shared title 
        fig.suptitle(f"Result of plot1() for {self.algo}", fontsize=14, fontweight='bold')  

        # Ensure plots don't overlap
        plt.tight_layout()
        
        # Save the plot
        current_directory = os.getcwd()
        save_path = f"{current_directory}\{self.algo}_convergence.png"
        print(save_path)
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Convergence plot saved at: {save_path}")
        return True


    def plot2(self):
        """ Compare E-greedy and Thompson Sampling cumulative rewards visually """
        
        # Initialize Epsilon Greedy and Thompson Sampling
        epsilon_algo = EpsilonGreedy(self.bandit_rewards, self.num_trials)
        thompson_algo = ThompsonSampling(self.bandit_rewards, self.num_trials)
        
        # Run the experiments and get the cumulative average rewards
        cumulative_rewards_epsilon = np.cumsum(epsilon_algo.experiment()[1])
        cumulative_rewards_thompson = np.cumsum(thompson_algo.experiment()[1])
        
        # Plotting the comparison
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Line Plot
        axes[0].plot(cumulative_rewards_epsilon, label='Epsilon Greedy')
        axes[0].plot(cumulative_rewards_thompson, label='Thompson Sampling')
        axes[0].set_xlabel('Number of Trials')
        axes[0].set_ylabel('Cumulative Reward')
        axes[0].set_title('Comparison of Cumulative Rewards: Line Plot')
        axes[0].legend()

        # Bar Chart
        x_labels = ['Epsilon Greedy', 'Thompson Sampling']
        x_pos = [0, 1]
        bar_width = 0.35
        bars = axes[1].bar(x_pos, [cumulative_rewards_epsilon[-1], cumulative_rewards_thompson[-1]], 
                     width=bar_width, tick_label=x_labels)
        axes[1].set_xlabel('Algorithm')
        axes[1].set_ylabel('Cumulative Reward')
        axes[1].set_title('Comparison of Cumulative Rewards: Bar Chart')

        # Adding numeric values on the bars with a frame
        for bar in bars:
            yval = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 2),
                          va='bottom', ha='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
       
        # Add a shared title 
        fig.suptitle("Result of plot2()", fontsize=14, fontweight='bold')  
    
        # Ensure plots don't overlap
        plt.tight_layout()

        # Save the plot
        current_directory = os.getcwd()
        save_path = f"{current_directory}\Reward_comparison.png"
        print(save_path)
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Cumulative rewards comparison plot saved at: {save_path}")
        return True
    

    def comparison(self):
        """ Compare the convergence of Thompson Sampling and Epsilon Greedy algorithms """
        
        # Initialize Thompson Sampling and Epsilon Greedy
        thompson_algo = ThompsonSampling(self.bandit_rewards, self.num_trials)
        epsilon_algo = EpsilonGreedy(self.bandit_rewards, self.num_trials)
        
        # Run the experiments and get the cumulative average rewards
        cumulative_rewards_thompson = thompson_algo.experiment()[0]
        cumulative_rewards_epsilon = epsilon_algo.experiment()[0]
        
        # Plotting the performance comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Linear scale plots
        axes[0].plot(cumulative_rewards_thompson, label='Thompson Sampling')
        axes[0].plot(cumulative_rewards_epsilon, label='Epsilon Greedy')
        axes[0].set_xlabel("Number of Trials")
        axes[0].set_ylabel("Cumulative Average Reward")
        axes[0].set_title("Comparison of Convergence: Thompson Sampling vs Epsilon Greedy (Linear Scale)", fontsize = 12)
        axes[0].legend()
        
        # Log scale plots
        axes[1].plot(cumulative_rewards_thompson, label='Thompson Sampling')
        axes[1].plot(cumulative_rewards_epsilon, label='Epsilon Greedy')
        axes[1].set_xlabel("Number of Trials")
        axes[1].set_ylabel("Cumulative Average Reward (Log Scale)")
        axes[1].set_title("Comparison of Convergence: Thompson Sampling vs Epsilon Greedy (Log Scale)", fontsize = 12)
        axes[1].set_xscale('log')
        axes[1].legend()

        # Add a shared title 
        fig.suptitle("Result of comparison()", fontsize=14, fontweight='bold', y=0.95)  

        # Ensure plots don't overlap
        plt.tight_layout(pad = 3)
        
        # Save the plot
        current_directory = os.getcwd()
        save_path = f"{current_directory}\Convergence_comparison.png"
        print(save_path)
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Convergence comparison plot saved at: {save_path}")
        return True        


#--------------------------------------#


class EBandit(Bandit):
    def __init__(self, m):
        self.m = m
        self.m_estimate = 0
        self.N = 0

    def __repr__(self):
        return f'A Bandit with {self.m} Win Rate'

    def pull(self):
        # Generate a random value from a normal distribution 
        return np.random.randn() + self.m

    def update(self, x):
        # Increment count of observations and estimate of the mean 
        self.N += 1
        self.m_estimate = ((1 - (1.0/self.N))*self.m_estimate) + ((1.0/self.N)*x)
        return('updated')

    def experiment(self):
        pass

    def report(self):
        pass
    

#--------------------------------------#


class EpsilonGreedy(Bandit): # + has a inheritance from EBandit
        def __init__(self, bandit_returns, N):
            self.bandit_returns = bandit_returns
            self.bandits = [EBandit(p) for p in self.bandit_returns]
            self.N = N
            
        def __repr__(self):
            return f'EpsilonGreedy algorithm with {len(self.bandits)} bandits\nRewards: {self.bandit_returns}'

        def pull(self):
            pass

        def update(self, x):
            pass

        def experiment(self):
            means = np.array(self.bandit_returns) # count number of suboptimal choices
            true_best = np.argmax(means)
            count_suboptimal = 0
            count_exploration = 0
            count_exploitation = 0
            data = np.empty(self.N)
            bandit_info = []

            for i in range(self.N):
                # Generate a random probability for exploration
                p = np.random.random()
                eps = 1/(i+1) # set a decaying eps
                if p < eps: # Exploration step
                    count_exploration += 1 # Choose increment exploration count and random bandit
                    j = np.random.choice(len(self.bandits))
                else: # Exploitation step
                    count_exploitation += 1  # Choose the bandit with the highest estimated mean reward
                    j = np.argmax([b.m_estimate for b in self.bandits])

                # Choose the bandit with the highest estimated mean reward
                x = self.bandits[j].pull()
                self.bandits[j].update(x)
                
                # Count suboptimal selections 
                if j != true_best:
                    count_suboptimal += 1

                # Store the reward obtained from this round and info about the chosen bandit
                data[i] = x
                bandit_info.append(repr(self.bandits[j]))
            
            # Calculate cumulative average rewards over time
            cumulative_average = np.cumsum(data) / (np.arange(self.N) + 1)

            # Get the estimated mean rewards for each bandit
            rewards_convergence=[b.m_estimate for b in self.bandits]
            return cumulative_average, data, rewards_convergence, bandit_info, \
                   count_suboptimal, count_exploration, count_exploitation

        def report(self):
            """
            Generates a report for the Epsilon Greedy algorithm including
            cumulative rewards, cumulative regret, and stores the rewards
            in a CSV file along with the bandit and algorithm information.
            """

            # Run the experiment to get cumulative rewards, rewards convergence, and bandit info
            cumulative_rewards, rewards, rewards_convergence,\
            bandit_info, count_suboptimal, count_exploration, count_exploitation = self.experiment()

            # Calculate cumulative regret
            optimal_reward = max(self.bandit_returns)
            cumulative_regret = np.sum(optimal_reward - np.array(rewards_convergence))

            # Log cumulative reward and cumulative regret
            logger.info(f"Cumulative Reward: {sum(rewards)}")
            logger.info(f"Cumulative Regret: {cumulative_regret}")
            logger.info(f"Number of times explored: {count_exploration}")
            logger.info(f"Number of times exploited: {count_exploitation}")
            logger.info(f"Number of suboptimal choices: {count_suboptimal}")

            # Prepare data for CSV
            data = {'Bandit': bandit_info,
                    'Reward': rewards,
                    'Algorithm': repr(self)}


            # Store data in CSV
            df = pd.DataFrame(data)
            current_directory = os.getcwd()
            save_path = f"{current_directory}/Report.csv"

            if os.path.isfile(save_path):
                # Append to existing CSV file without header
                df.to_csv(save_path, mode='a', header=False, index=False)
            else:
                # Create a new CSV file and write the data
                df.to_csv(save_path, index=False)


            # Visualization
            visualizer = Visualization('EpsilonGreedy', self.bandit_returns, self.N)

            # Call plot1 to visualize the learning process
            visualizer.plot1()

            # Call plot2 to compare cumulative rewards of this algo with similar Thompson
            visualizer.plot2()

            # Call comparison to compare convergence of this algo with similar Thompson
            visualizer.comparison()

            logger.info(f"Report generated and rewards saved as {save_path}")
            return True
        


#--------------------------------------#


class TBandit(Bandit):
    def __init__(self, true_mean):
        self.true_mean = true_mean
        # parameters for mu - prior is N(0,1)
        self.m = 0
        self.lambda_ = 1
        self.tau = 1
        self.N = 0
        self.sum_x = 0
    
    def __repr__(self):
        return f'A Bandit with {self.true_mean} Win Rate'

    def pull(self):
        # Generate a random value from a normal distribution 
        return np.random.randn() / np.sqrt(self.tau) + self.true_mean
    
    def update(self, x):
        # Increment lambda by tau
        self.lambda_ += self.tau
        # Update the sum of observed values
        self.sum_x += x
        # Update the estimated
        self.m = (self.tau * self.sum_x) / self.lambda_
        # Increment the count of trials
        self.N += 1

    def experiment(self):
        pass

    def report(self):
        pass

    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda_) + self.m


#--------------------------------------#


class ThompsonSampling(Bandit): # + has a inheritance from TBandit
    def __init__(self, bandit_means, N):
            self.bandit_means = bandit_means
            self.bandits = [TBandit(m) for m in self.bandit_means]
            self.N = N
            

    def __repr__(self):
        return f'ThompsonSampling algorithm with {len(self.bandits)} bandits\nRewards: {self.bandit_means}'
    
    def pull(self):
        pass

    def update(self):
        pass

    def experiment(self):
        # Create bandits based on given means
        # Initialize arrays and variables to track rewards, bandit information, and suboptimal selections
        bandits = [TBandit(m) for m in self.bandit_means]
        rewards = np.empty(self.N)
        bandit_info = []
        count_suboptimal = 0

        # Run the experiment for N rounds
        for i in range(self.N):
            # Choose the bandit with the highest sample value
            j = np.argmax([b.sample() for b in bandits])
             
            # Pull the chosen bandit's arm and update its state
            x = bandits[j].pull()
            bandits[j].update(x)

            # Record the reward obtained and information about the chosen bandit
            rewards[i] = x
            bandit_info.append(repr(bandits[j]))

            # Check if the chosen bandit is suboptimal
            if j != np.argmax(self.bandit_means):  
                count_suboptimal += 1

        # Calculate cumulative average rewards
        cumulative_average = np.cumsum(rewards) / (np.arange(self.N) + 1)       
        # Get the estimated means of each bandit after the experiment                              
        reward_convergence = [b.m for b in bandits]
        return cumulative_average, rewards, reward_convergence, bandit_info, count_suboptimal

    
    def report(self):
        """
        Generates a report for the Thompson Sampling algorithm including
        cumulative rewards, cumulative regret, and stores the rewards
        in a CSV file along with the bandit and algorithm information.
        """

        # Run the experiment to get cumulative rewards, rewards convergence, and bandit info
        cumulative_rewards, rewards, rewards_convergence, bandit_info, count_suboptimal = self.experiment()

        # Calculate cumulative regret
        optimal_reward = max(self.bandit_means)
        cumulative_regret = np.sum(optimal_reward - np.array(rewards_convergence))

        # Log cumulative reward and cumulative regret
        logger.info(f"Cumulative Reward: {sum(rewards)}")
        logger.info(f"Cumulative Regret: {cumulative_regret}")
        logger.info(f"Number of suboptimal choices: {count_suboptimal}")

        # Prepare data for CSV
        data = {'Bandit': bandit_info,
                'Reward': rewards,
                'Algorithm': repr(self)}


        # Store data in CSV
        df = pd.DataFrame(data)
        current_directory = os.getcwd()
        save_path = f"{current_directory}/Report.csv"

        if os.path.isfile(save_path):
            # Append to existing CSV file without header
            df.to_csv(save_path, mode='a', header=False, index=False)
        else:
            # Create a new CSV file and write the data
            df.to_csv(save_path, index=False)


        # Visualization
        visualizer = Visualization('ThompsonSampling', self.bandit_means, self.N)

        # Call plot1 to visualize the learning process
        visualizer.plot1()

        # Call plot2 to compare cumulative rewards with similar EpsilonGreedy
        visualizer.plot2()

        # Call comparison to compare convergence with similar EpsilonGreedy
        visualizer.comparison()

        logger.info(f"Report generated and rewards saved as {save_path}")
        return True
    

#--------------------------------------#


if __name__=='__main__':

    # Define your Bandit_Reward and NumberOfTrials 
    Bandit_Reward = [1, 2, 3, 4]
    NumberOfTrials = 20000
    ALGORITHMS = ['EpsilonGreedy', 'ThompsonSampling']
    
    """     
    # Initialize the Visualization object
    visualizerE = Visualization(ALGORITHMS[0], Bandit_Reward, NumberOfTrials)
    visualizerT = Visualization(ALGORITHMS[1], Bandit_Reward, NumberOfTrials)

    # Plot the performance for EpsilonGreedy and log the return statement
    visualizerE.plot1()

    # Plot the performance for ThompsonSampling and log the return statement
    visualizerT.plot1()

    # Comparisona and log the return statements
    visualizerE.plot2() # compare rewards
    visualizerT.plot2() # same as above

    visualizerE.comparison() # compare convergences
    visualizerT.comparison() # same as above """
    
    # Below lines will run everything and generate full report including visuals in separate .png files
    ThompsonSampling(Bandit_Reward, NumberOfTrials).report()
    EpsilonGreedy(Bandit_Reward, NumberOfTrials).report()


#--------------------------------------#


###### ANOTHER IMPLEMENTATION PLAN ######

""" 
Bandit(ABC) with __repr__, update, pull  methods
BanditE(Bandit)
BanditT(Bandit)

Algorithm(ABC) with __reprt__, experiment  methods
AlgorithmE(Algorithm)
AlgorithmT(Algorithm)

Report(ABC) with generate_full_report, plot1, plo2, comparison  methods
MyReport(Report)
"""

# this way we wouldn't have methods inside classes that 
# wouldn't be used and just passed + the logical flow would be easier