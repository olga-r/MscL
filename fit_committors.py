import numpy as np
from scipy.optimize import minimize
from itertools import combinations
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from matplotlib import pyplot as plt
from copy import deepcopy 
from matplotlib.figure import Figure
import matplotlib as mpl
import sys
import os
import glob, re 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.utils import resample
import argparse
import random
from scipy.stats import ttest_1samp



parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, help="random integer")
args = parser.parse_args()


random_state = args.seed
def y(theta, t):
    return 1 / (1 + np.exp(- theta[0] - theta[1]*t[0]- theta[2]*t[1]- theta[3]*t[2]))


def y2(theta, t):
    a0, *an = theta
    r = a0
    for e, a in enumerate(an):
        r += a*t[e]   
    return 1 / (1 + np.exp(-r))

def fun(theta):
    return y2(theta, ts_train) - ys_train

def calculate_bic(n, mse, num_params):
    bic = n * np.log(mse) + num_params * np.log(n)
    return bic

def calculate_aic(n, mse, num_params):
    aic = n * np.log(mse) + 2 * num_params
    return aic

def monte_carlo(old_value, new_value):
    return min(1, np.exp(500*(new_value-old_value)))

def r2_score(y_true, y_pred):
    ssr = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ssr / sst

def adjusted_r2_score(y_true, y_pred, n_points,n_features):
    n, k = n_points,n_features
    ssr = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ssr / sst
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


names = ["{}m2".format(i) for i in range(1,116)]


data = np.loadtxt("features_and_committors.dat")[:,0:-1]
probs = np.loadtxt("features_and_committors.dat")[:, -1]


data_0 = data[probs==0]
data_1 = data[probs==1]
data_p1 = data[(probs>0)&(probs<0.1)]
data_p2 = data[(probs>=0.1)&(probs<0.25)]
data_p3 = data[(probs>=0.25)&(probs<0.4)]
data_p4 = data[(probs>=0.4)&(probs<=0.6)]
data_p5 = data[(probs>0.6)&(probs<=0.75)]
data_p6 = data[(probs>0.75)&(probs<=0.9)]
data_p7 = data[(probs>0.9)&(probs< 1)]
probs_0 = probs[probs==0]
probs_1 = probs[probs==1]
probs_p1 = probs[(probs>0)&(probs<0.1)]
probs_p2 = probs[(probs>=0.1)&(probs<0.25)]
probs_p3 = probs[(probs>=0.25)&(probs<0.4)]
probs_p4 = probs[(probs>=0.4)&(probs<=0.6)]
probs_p5 = probs[(probs>0.6)&(probs<=0.75)]
probs_p6 = probs[(probs>0.75)&(probs<=0.9)]
probs_p7 = probs[(probs>0.9)&(probs< 1)]
print(len(probs_0),len(probs_p1),len(probs_p2),len(probs_p3),len(probs_p4),len(probs_p5),len(probs_p6),len(probs_p7),len(probs_1) )
#96 55 54 41 61 62 79 176 596

q_0_t, q_0_test, p_0_t, p_0_test = train_test_split(data_0, probs_0,test_size=0.1, shuffle=True, random_state=random_state)
q_1_t, q_1_test, p_1_t, p_1_test = train_test_split(data_1, probs_1,test_size=0.03, shuffle=True, random_state=random_state)
q_p1_t, q_p1_test, p_p1_t, p_p1_test = train_test_split(data_p1, probs_p1,test_size=0.16, shuffle=True, random_state=random_state)
q_p2_t, q_p2_test, p_p2_t, p_p2_test = train_test_split(data_p2, probs_p2,test_size=0.16, shuffle=True, random_state=random_state)
q_p3_t, q_p3_test, p_p3_t, p_p3_test = train_test_split(data_p3, probs_p3,test_size=0.15, shuffle=True, random_state=random_state)
q_p4_t, q_p4_test, p_p4_t, p_p4_test = train_test_split(data_p4, probs_p4,test_size=0.14, shuffle=True, random_state=random_state)
q_p5_t, q_p5_test, p_p5_t, p_p5_test = train_test_split(data_p5, probs_p5,test_size=0.14, shuffle=True, random_state=random_state)
q_p6_t, q_p6_test, p_p6_t, p_p6_test = train_test_split(data_p6, probs_p6,test_size=0.12, shuffle=True, random_state=random_state)
q_p7_t, q_p7_test, p_p7_t, p_p7_test = train_test_split(data_p7, probs_p7,test_size=0.06, shuffle=True, random_state=random_state)
print(len(p_0_test),len(p_p1_test),len(p_p2_test),len(p_p3_test),len(p_p4_test),len(p_p5_test),len(p_p6_test),len(p_p7_test),len(p_1_test) )

q_test = np.concatenate((q_0_test, q_p1_test, q_p2_test, q_p3_test, q_p4_test, q_p5_test, q_p6_test, q_p7_test, q_1_test), axis=0)
p_test = np.concatenate((p_0_test, p_p1_test, p_p2_test,  p_p3_test,  p_p4_test,  p_p5_test,  p_p6_test,  p_p7_test, p_1_test), axis=0)
data_test = np.c_[q_test, p_test]
q_t = np.concatenate((q_0_t, q_p1_t, q_p2_t, q_p3_t, q_p4_t, q_p5_t, q_p6_t, q_p7_t, q_1_t), axis=0)
p_t = np.concatenate((p_0_t, p_p1_t, p_p2_t,  p_p3_t,  p_p4_t,  p_p5_t,  p_p6_t,  p_p7_t, p_1_t), axis=0)
data_other = np.c_[q_t, p_t]
np.savetxt("data_test_{}".format(random_state), data_test, fmt='%10.6f')
np.savetxt("data_train_{}".format(random_state), data_other, fmt='%10.6f')



datasets= {}
rnds = random.sample(range(0, 2**32), 1)
count = 0
for ernd, rnd in enumerate(rnds):
    q_0_train, q_0_valid, p_0_train, p_0_valid = train_test_split(q_0_t, p_0_t,test_size=0.1, shuffle=True, random_state=rnd)
    q_1_train, q_1_valid, p_1_train, p_1_valid = train_test_split(q_1_t, p_1_t,test_size=0.02, shuffle=True, random_state=rnd)
    q_p1_train, q_p1_valid, p_p1_train, p_p1_valid = train_test_split(q_p1_t,  p_p1_t,test_size=0.2, shuffle=True, random_state=rnd)
    q_p2_train, q_p2_valid, p_p2_train, p_p2_valid = train_test_split(q_p2_t,  p_p2_t,test_size=0.2, shuffle=True, random_state=rnd)
    q_p3_train, q_p3_valid, p_p3_train, p_p3_valid = train_test_split(q_p3_t,  p_p3_t,test_size=0.2, shuffle=True, random_state=rnd)
    q_p4_train, q_p4_valid, p_p4_train, p_p4_valid = train_test_split(q_p4_t,  p_p4_t,test_size=0.2, shuffle=True, random_state=rnd)
    q_p5_train, q_p5_valid, p_p5_train, p_p5_valid = train_test_split(q_p5_t,  p_p5_t,test_size=0.2, shuffle=True, random_state=rnd)
    q_p6_train, q_p6_valid, p_p6_train, p_p6_valid = train_test_split(q_p6_t,  p_p6_t,test_size=0.15, shuffle=True, random_state=rnd)
    q_p7_train, q_p7_valid, p_p7_train, p_p7_valid = train_test_split(q_p7_t,  p_p7_t,test_size=0.06, shuffle=True, random_state=rnd)
    print(len(p_0_valid),len(p_p1_valid),len(p_p2_valid),len(p_p3_valid),len(p_p4_valid),len(p_p5_valid),len(p_p6_valid),len(p_p7_valid),len(p_1_valid) )
    q_valid = np.concatenate((q_0_valid, q_p1_valid, q_p2_valid, q_p3_valid, q_p4_valid, q_p5_valid, q_p6_valid, q_p7_valid, q_1_valid), axis=0)
    p_valid = np.concatenate((p_0_valid, p_p1_valid, p_p2_valid,  p_p3_valid,  p_p4_valid,  p_p5_valid,  p_p6_valid,  p_p7_valid, p_1_valid), axis=0)

    for bts in range(20):
        q_0_train2, p_0_train2 = resample(q_0_train, p_0_train, n_samples=70)
        q_1_train2, p_1_train2 = resample(q_1_train, p_1_train, n_samples=70)
        q_p1_train2, p_p1_train2 = resample(q_p1_train, p_p1_train, n_samples=70)
        q_p2_train2, p_p2_train2 = resample(q_p2_train, p_p2_train, n_samples=70)
        q_p3_train2, p_p3_train2 = resample(q_p3_train, p_p3_train, n_samples=70)
        q_p4_train2, p_p4_train2 = resample(q_p4_train, p_p4_train, n_samples=70)
        q_p5_train2, p_p5_train2 = resample(q_p5_train, p_p5_train, n_samples=70)
        q_p6_train2, p_p6_train2 = resample(q_p6_train, p_p6_train, n_samples=70)
        q_p7_train2, p_p7_train2 = resample(q_p7_train, p_p7_train, n_samples=70)

        q_train = np.concatenate((q_0_train2, q_p1_train2, q_p2_train2, q_p3_train2, q_p4_train2, q_p5_train2, q_p6_train2, q_p7_train2, q_1_train2), axis=0)
        p_train = np.concatenate((p_0_train2, p_p1_train2, p_p2_train2,  p_p3_train2,  p_p4_train2,  p_p5_train2,  p_p6_train2,  p_p7_train2, p_1_train2), axis=0)

        datasets[count] = {"q_valid": q_valid, "p_valid": p_valid, "q_train": q_train, "p_train": p_train}
        count += 1



selected_features =['1m2', "8m2", "12m2","14m2", "20m2","33m2","42m2", "4m2","54m2","60m2","64m2","67m2","3m2","82m2","85m2","87m2","90m2","92m2","95m2","98m2","102m2"]
rest_features = list(set(["{}m2".format(i) for i in range(1,116)])-set(selected_features))
idx = []
for feat in selected_features:
    idx.append(names.index(feat))

max_p_threshold = 0.05
coeffs = None
step = 0
best_model = {"r2": 0, "mean_mse": 0, "std_mse": 0, "features": [], "coeffs": []}
r2_to_save = 0
r2_old =0

if len(selected_features) > 0:
    dict_keys = ["a0"]+selected_features
    result = {}
    for k in dict_keys:
        result[k] = []
    result["mse"] = []
    result["r2"] = []
    for count in range(20):
        q_train, q_valid, p_train,p_valid = datasets[count]["q_train"],datasets[count]["q_valid"],datasets[count]["p_train"],datasets[count]["p_valid"]
        start_params = [ 1]
        ts_train,ts_valid = [],[]
        for ii in idx:
            ts_train.append(q_train[:, ii])
            ts_valid.append(q_valid[:, ii])
            start_params.append(1)
        ys_train = p_train
        ys_valid = p_valid
        res = least_squares(fun, start_params, loss='soft_l1')
        result["mse"].append(mean_squared_error(y2(res.x, ts_valid),ys_valid))
        result["r2"].append(adjusted_r2_score(ys_valid, y2(res.x, ts_valid),len(ys_valid), len(selected_features)))
        for n,coeff in zip(dict_keys,res.x):
            result[n].append(coeff)
    mse = np.array(result["mse"])
    r2 = np.mean(np.array(result["r2"]))
    r2_to_save = r2
    best_model["r2"] = r2
    best_model["mean_mse"] = np.mean(mse)
    best_model["std_mse"] = np.std(result["mse"])
    best_model["features"] = deepcopy(selected_features)
    best_model["coeffs"] = deepcopy(result)
    r2_old=r2
    print(r2_old)
    #print("initial R2 adjusted: {}".format(r2))
    
    


while step < 1000:
    ACCEPT = False
    CHECK = False
    #print("Step {}".format(step))
    step+=1    
    total_results = []

    #Monte Carlo
    feat = random.choice(rest_features)
    i = names.index(feat)
    idx_temp = deepcopy(idx)
    idx_temp.append(i)
    temp_features = deepcopy(selected_features)
    temp_features.append(feat)
    dict_keys = ["a0"]+temp_features
    result = {}
    for k in dict_keys:
        result[k] = []
    result["mse"] = []
    result["r2"] = []
    for count in range(20):
        q_train, q_valid, p_train,p_valid = datasets[count]["q_train"],datasets[count]["q_valid"],datasets[count]["p_train"],datasets[count]["p_valid"]
        start_params = [ 1]
        ts_train,ts_valid = [],[]
        for ii in idx_temp:
            ts_train.append(q_train[:, ii])
            ts_valid.append(q_valid[:, ii])
            start_params.append(1)
        ys_train = p_train
        ys_valid = p_valid
        res = least_squares(fun, start_params, loss='soft_l1')
        result["mse"].append(mean_squared_error(y2(res.x, ts_valid),ys_valid))
        result["r2"].append(adjusted_r2_score(ys_valid, y2(res.x, ts_valid),len(ys_valid), len(temp_features)))
        for n,coeff in zip(dict_keys,res.x):
            result[n].append(coeff)
    mse = np.mean(np.array(result["mse"]))
    r2_new = np.mean(np.array(result["r2"]))
    p_accept = monte_carlo(r2_old, r2_new)
    p_random = random.random()
    #print(feat, p_accept, p_random)
    if p_random<=p_accept:
        selected_features.append(feat)
        rest_features.remove(feat)
        idx.append(names.index(feat)) 
        r2_old = r2_new
        ACCEPT=True
        #print("Forward: add {}. Resulted MSE is: {} (std = {}). Resulted R2 adjusted is: {}".format(feat, mse,np.std(result["mse"]), r2_new))
        coeffs = result
        if r2_new > r2_to_save:
            r2_to_save = r2_new
            best_model["r2"] = r2_new
            best_model["mean_mse"] = mse
            best_model["std_mse"] = np.std(result["mse"])
            best_model["features"] = deepcopy(selected_features)
            best_model["coeffs"] = deepcopy(coeffs)


    #check
    if len(selected_features)> 1 and ACCEPT:
        max_p = 0
        worse_feat = None
        
        for k, v in result.items():
            if k in ["a0", "mse","r2"]:
                continue
            stat, p_value = ttest_1samp(v, popmean=0)
            if p_value > max_p:
                max_p = p_value
                worse_feat = k
        if max_p >= max_p_threshold:
            selected_features.remove(worse_feat)
            rest_features.append(worse_feat)
            idx.remove(names.index(worse_feat))
            #print("remove {} with p_val: {}".format(worse_feat, max_p))
            CHECK=True
            while True:
                dict_keys = ["a0"]+selected_features
                #print(dict_keys)
                result = {}
                for k in dict_keys:
                    result[k] = []
                result["mse"] = []
                result["r2"] = []
                for count in range(20):
                    q_train, q_valid, p_train,p_valid = datasets[count]["q_train"],datasets[count]["q_valid"],datasets[count]["p_train"],datasets[count]["p_valid"]
                    start_params = [ 1]
                    ts_train,ts_valid = [],[]
                    for ii in idx:
                        ts_train.append(q_train[:, ii])
                        ts_valid.append(q_valid[:, ii])
                        start_params.append(1)
                    ys_train = p_train
                    ys_valid = p_valid
                    res = least_squares(fun, start_params, loss='soft_l1')
                    result["mse"].append(mean_squared_error(y2(res.x, ts_valid),ys_valid))
                    result["r2"].append(adjusted_r2_score(ys_valid, y2(res.x, ts_valid),len(ys_valid), len(selected_features)))
                    for n,coeff in zip(dict_keys,res.x):
                        result[n].append(coeff)

                max_p = 0
                for k, v in result.items():
                    if k in ["a0", "mse","r2"]:
                        continue
                    stat, p_value = ttest_1samp(v, popmean=0)
                    if p_value > max_p:
                        max_p = p_value
                        worse_feat = k
                if max_p >= max_p_threshold:
                    selected_features.remove(worse_feat)
                    rest_features.append(worse_feat)
                    idx.remove(names.index(worse_feat))
                    #print("remove {} with p_val: {}".format(worse_feat, max_p))
                else:
                    break
            mse = np.array(result["mse"])
            r2_new = np.mean(np.array(result["r2"]))
            r2_old=r2_new
            #print("Check removed: Now there are {} features. Resulted MSE is: {} (std = {}). Resulted R2 adjusted is: {}".format(len(selected_features), np.mean(mse),np.std(mse), r2_new))
            if r2_new > r2_to_save:
                r2_to_save = r2_new
                best_model["r2"] = r2_new
                best_model["mean_mse"] = np.mean(mse)
                best_model["std_mse"] = np.std(result["mse"])
                best_model["features"] = deepcopy(selected_features)
                best_model["coeffs"] = deepcopy(result)
   


    
    ACCEPT=False
    if len(selected_features)> 1 and not CHECK:
        feat = random.choice(selected_features)
        i  = names.index(feat)
        idx_temp = deepcopy(idx)
        idx_temp.remove(i)
        temp_features = deepcopy(selected_features)
        temp_features.remove(feat)
        dict_keys = ["a0"]+temp_features
        result = {}
        for k in dict_keys:
            result[k] = []
        result["mse"] = []
        result["r2"] = []    
        for count in range(20):
            q_train, q_valid, p_train,p_valid = datasets[count]["q_train"],datasets[count]["q_valid"],datasets[count]["p_train"],datasets[count]["p_valid"]
            start_params = [ 1]
            ts_train,ts_valid = [],[]
            for ii in idx_temp:
                ts_train.append(q_train[:, ii])
                ts_valid.append(q_valid[:, ii])
                start_params.append(1)
            ys_train = p_train
            ys_valid = p_valid
            res = least_squares(fun, start_params, loss='soft_l1')
            result["mse"].append(mean_squared_error(y2(res.x, ts_valid),ys_valid))
            result["r2"].append(adjusted_r2_score(ys_valid, y2(res.x, ts_valid),len(ys_valid), len(temp_features)))
            for n,coeff in zip(dict_keys,res.x):
                result[n].append(coeff)
        mse = np.mean(np.array(result["mse"]))
        r2_new = np.mean(np.array(result["r2"]))
        p_accept = monte_carlo(r2_old, r2_new)
        p_random = random.random()
        #print(feat, p_accept, p_random)
        if p_random<=p_accept:
            selected_features.remove(feat)
            rest_features.append(feat)
            idx.remove(names.index(feat)) 
            r2_old = r2_new
            ACCEPT=True
            #print("Backward: remove {}. Resulted MSE is: {} (std = {}). Resulted R2 adjusted is: {}".format(feat, mse,np.std(result["mse"]), r2_new))
            coeffs = result
            if r2_new > r2_to_save:
                r2_to_save = r2_new
                best_model["r2"] = r2_new
                best_model["mean_mse"] = mse
                best_model["std_mse"] = np.std(result["mse"])
                best_model["features"] = deepcopy(selected_features)
                best_model["coeffs"] = deepcopy(coeffs)
        
    #check
    if len(selected_features)> 1 and ACCEPT:
        max_p = 0
        worse_feat = None
        
        for k, v in result.items():
            if k in ["a0", "mse","r2"]:
                continue
            stat, p_value = ttest_1samp(v, popmean=0)
            if p_value > max_p:
                max_p = p_value
                worse_feat = k
        if max_p >= max_p_threshold:
            selected_features.remove(worse_feat)
            rest_features.append(worse_feat)
            idx.remove(names.index(worse_feat))
            #print("remove {} with p_val: {}".format(worse_feat, max_p))
            CHECK=True
            while True:
                dict_keys = ["a0"]+selected_features
                #print(dict_keys)
                result = {}
                for k in dict_keys:
                    result[k] = []
                result["mse"] = []
                result["r2"] = []
                for count in range(20):
                    q_train, q_valid, p_train,p_valid = datasets[count]["q_train"],datasets[count]["q_valid"],datasets[count]["p_train"],datasets[count]["p_valid"]
                    start_params = [ 1]
                    ts_train,ts_valid = [],[]
                    for ii in idx:
                        ts_train.append(q_train[:, ii])
                        ts_valid.append(q_valid[:, ii])
                        start_params.append(1)
                    ys_train = p_train
                    ys_valid = p_valid
                    res = least_squares(fun, start_params, loss='soft_l1')
                    result["mse"].append(mean_squared_error(y2(res.x, ts_valid),ys_valid))
                    result["r2"].append(adjusted_r2_score(ys_valid, y2(res.x, ts_valid),len(ys_valid), len(selected_features)))
                    for n,coeff in zip(dict_keys,res.x):
                        result[n].append(coeff)

                max_p = 0
                for k, v in result.items():
                    if k in ["a0", "mse","r2"]:
                        continue
                    stat, p_value = ttest_1samp(v, popmean=0)
                    if p_value > max_p:
                        max_p = p_value
                        worse_feat = k
                if max_p >= max_p_threshold:
                    selected_features.remove(worse_feat)
                    rest_features.append(worse_feat)
                    idx.remove(names.index(worse_feat))
                    #print("remove {} with p_val: {}".format(worse_feat, max_p))
                else:
                    break
            mse = np.array(result["mse"])
            r2_new = np.mean(np.array(result["r2"]))
            r2_old=r2_new
            #print("Check removed: Now there are {} features. Resulted MSE is: {} (std = {}). Resulted R2 adjusted is: {}".format(len(selected_features), np.mean(mse),np.std(mse), r2_new))
            if r2_new > r2_to_save:
                r2_to_save = r2_new
                best_model["r2"] = r2_new
                best_model["mean_mse"] = np.mean(mse)
                best_model["std_mse"] = np.std(result["mse"])
                best_model["features"] = deepcopy(selected_features)
                best_model["coeffs"] = deepcopy(result)
   


print(best_model["features"])
print("R2 {}".format(best_model["r2"]))
print("MSE {} (std:  {})".format(best_model["mean_mse"], best_model["std_mse"]))


for k, v in best_model["coeffs"].items():

    stat, p_value = ttest_1samp(v, popmean=0)
    v = np.array(v)
    print(k, np.mean(v), np.std(v), stat, p_value)

    
    


    
