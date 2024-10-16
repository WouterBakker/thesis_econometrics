# Thesis progress
### On this webpage, there is an overview of what I've done so far, with links to the corresponding Jupyter notebooks. The notebooks contain information on what I've implemented and how the code works.

17-10-2024, 13:17

- General
	- My functions are implemented in the file [class_FE_regression.py](Python/class_FE_regression.py)
		- When a class or function is called in a notebook, it will be in this file

1. [Simulation study: Fixed Effects regression](Python/FE_OLS_hypothesis_testing.html)
	1. Implemented so far: simulation, estimation, t-values, type I errors
	2. Two models are compared in the notebook:
		1. Regular FE model
			1. Regular FE model estimated using the within (demeaning) transformation
		2. Weighted FE model
			1. Accounts for missing values in X
			2. Missing values occur for two reasons in pseudo-panel models:
				1. General missingness: no observations for a time-period since survey wasn't taken in that year for specific countries
				2. Cohort missingness: some cohorts were not observed
				3. Item missingness: a respondent declined to answer a specific question
			3. Procedure
				1. Sets some observations to zero (to simulate missingness)
				2. Constructs a weighting matrix based these missing values so they are omitted in estimation (0-values don't affect sum of squares)
				3. Corrects the variance calculation used to construct the covariance matrix
					1. One of the steps is to calculate the variance for each cross-sectional unit
					2. Normally we can divide by T, but since T is different for each unit this needs to be accounted for (more details in the notebook)
2. [Replication of Inoue (2008): "efficient estimation and inference in linear pseudo-panel data models"](Python/Inoue_replication.html)
	- Pdf: [Inoue (2008): "efficient estimation and inference in linear pseudo-panel data models"](Papers/Inoue_efficient_estimation.pdf)
	1. Inoue suggests a GMM estimator based on feasible GLS that is more efficient than the standard FE estimator for pseudo-panel models
	2. He performs a Monte Carlo simulation study and finds that the RMSE of $\beta$ is much lower for the GMM estimator as compared to the FE estimator
		1. My RMSE is indeed significantly lower, but the results are not the same as in Inoue, and I'm unsure why
		2. Need to verify whether my **simulation parameters** are indeed the same as Inoue's
			1. It might be that I'm simulating the data incorrectly.
3. [European Social Survey (ESS) analysis](Python/ESS_pseudo_panel_analysis.html)
	1. The European Social Survey (ESS) is a cross-national survey conducted biennially across Europe, collecting data on social, political, and behavioral attitudes and values. It aims to track long-term changes and differences across countries, providing insights for social science research and informing policy decisions.
	2. Good candidate for an analysis, since the documentation is very good. 
	3. The aim (currently) is to provide methodological guidelines on how one might approach the analysis of survey data using the proposed pseudo-panel model.
		- Constructing the cohorts based on post-stratification
			- Need to consider the instrumental variabel interpretation of cohort formation:
				- Exogeneity and relevance conditions
		-  Applying the weights correctly
		- Informed choice for imputation techniques when data are Missing at Random (as opposed to Missing Completely at Random, which is ignoreable)
	4. [Distribution of variables](Python/ESS_bar_charts.html)
		- Most questions in the survey are scored on a 10-point scale. For certain variables of interest, I plot their distribution and show item missingness. 
