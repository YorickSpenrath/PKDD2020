# PKDD2020

Welcome! You have found the code to my ECML-PKDD submission.

This readme should get you started with running the code and recreating the experiments from scratch. As this is a
Python project; I recommend creating a virtual environment. Once done, run the following from the repo directory:

```pip install -r requirements.txt```

That should set up the environment with all packages required to recompute the results. The results can be computed per
each of the three experiments; the scripts for these are in scripts\PKDD_EXPERIMENTS. Each experiment contains two 
scripts; one for "COMPUTE" and one "RESULTS". As suggested by the name, "COMPUTE" will do all of the computations. The 
"RESULTS" script on the other hand will format them in a nice, presentable manner (it will generally create a Tex file 
containting a table which can be input in another Tex file.) Apart from these, there are the ```ALL_EXPERIMENTS.py```
and ```POST_BOOTSTRAP.py``` scripts. These will allow you to run all scripts at once. Mind that some of these take some
time; especially the computation of the first experiment. The latter of these two will actually skip this first 
computation and use the precomputed results, located in ```RESULTS\Precomputed```.

By default, the experiment uses 1000 repetitions for each dataset (as per the paper), but these settings can be changed
in the ```PKDD_PARAMETERS``` file. Results of experiment 1 have been precomputed for ```100``` and ```1000```
repetitions.

Executing the scripts (```ALL_EXPERIMENTS``` or ```POST_BOOTSTRAP```) will create the folder
```RESULTS\PKDD 1000 Reps``` which contains all of the results.

As this is part of an ongoing project:

- There will be several TODO's scattered over the repository. These do not break the code, nor do they invalidate the
results.
- This code will likely become outdated at some point; in the sense that we have developed newer, better and/or extended
versions. If and when this happens; a link will be provided here. 