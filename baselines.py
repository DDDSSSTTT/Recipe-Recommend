import gzip
import os
import pickle
import csv
import numpy as np
from math import sqrt
from statistics import median,mean
from collections import defaultdict
from sklearn import linear_model
import random
my_seed = 2338
random.seed(my_seed)
np.random.seed(my_seed)

def readGz(path):
    for l in gzip.open(path, 'rt',encoding='utf-8'):
        yield eval(l)

def readCSV(path):
    f = gzip.open(path, 'rt')
    c = csv.reader(f)
    header = next(c)
    for l in c:
        d = dict(zip(header,l))
        yield d['user_id'],d['recipe_id'],d

### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

allRatings = []
userRatings = defaultdict(list)
trainset = []
validset = []
idx=0
for user,recipe,d in readCSV("trainInteractions.csv.gz"):
  r = int(d['rating'])
  allRatings.append(r)
  userRatings[user].append(r)
  if idx<400000:
    trainset.append((user,recipe,d))
  else:
    validset.append((user,recipe,d))
  idx += 1


globalAverage = sum(allRatings) / len(allRatings)
userAverage = {}
for u in userRatings:
  userAverage[u] = sum(userRatings[u]) / len(userRatings[u])

from datetime import datetime
from datetime import timedelta
time_now=datetime.strptime('2021-09-01','%Y-%m-%d')
def parse_date(t):
  date = t
  parsed_time = datetime.strptime(date, '%Y-%m-%d')
  return (time_now-parsed_time).days//30
def parse_date_u(u):
  if len(userDate[u])==0:
    result = AverageMonths
  else:
    result = median(userDate[u])
  return result
def parse_date_i(i):
  if len(recipeDate[i])==0:
    result = AverageMonths
  else:
    result = median(recipeDate[i])
  return result
### Would-cook baseline: just rank which recipes are popular and which are not, and return '1' if a recipe is among the top-ranked

recipeCount = defaultdict(int)
userCoocked=defaultdict(list)
recipeCoocked=defaultdict(list)
userDate = defaultdict(list)
recipeDate = defaultdict(list)
totalCooked = 0
totalDays = 0
for user,recipe,d in trainset:
  recipeCount[recipe] += 1
  userCoocked[user].append(recipe)
  recipeCoocked[recipe].append(user)
  imonths=parse_date(d['date'])
  userDate[user].append(imonths)
  recipeDate[recipe].append(imonths)
  totalDays +=imonths
  totalCooked += 1

AverageMonths= totalDays / totalCooked

mostPopular = [(recipeCount[x], x) for x in recipeCount]
mostPopular.sort()
mostPopular.reverse()
mostPopularUser = [(len(userCoocked[x]),x) for x in userCoocked]
mostPopularUser.sort()
mostPopularUser.reverse()
# Q1: Cut the pop set
def pop_item_cut(percentage=0.5):
  return1 = set()
  count = 0
  for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count >= totalCooked*percentage: break #choose top 50%
  return return1

def pop_user_cut(percentage=0.5):
  return2 = set()
  count = 0
  for uc, u in mostPopularUser:
    count += uc
    return2.add(u)
    if count >= totalCooked*percentage: break #choose top 50%
  return return2

return1 = pop_item_cut(0.625)
return2 = pop_user_cut(0.25)

def in_return1(i):
  return (i in return1)
def in_return2(u):
  return (u in return1)

start_date = datetime.strptime('2007-01-01','%Y-%m-%d')
end_date = datetime.strptime('2021-09-01','%Y-%m-%d')
time_between_dates = end_date - start_date
days_between_dates = time_between_dates.days
def randomdate():
  random_number_of_days = random.randrange(days_between_dates)
  random_date = start_date + timedelta(days=random_number_of_days)
  return random_date

#Validation Part
#1. Rebuild train, validation set
new_train_set=[i[:2]+(i[2]['date'],)+(1,) for i in trainset]
new_validation_set=[i[:2]+(i[2]['date'],)+(1,) for i in validset]
# new_train_set=[i[:2]+(1,) for i in trainset]
# new_validation_set=[i[:2]+(1,) for i in validset]
negative_train_set=[]
negative_validation_set=[]
import random
recipes_=list(recipeCount.keys())
least_accessed = [d[1] for d in mostPopular[int(0.05 * len(mostPopular)):]]
for each in new_train_set:
  x=random.choice(least_accessed)
  user=each[0]
  date=datetime.strftime(randomdate(),'%Y-%m-%d')
  while x in userCoocked[user]:
    x = random.choice(least_accessed)
  negative_train_set.append((user,x,date,0))
  # negative_train_set.append((user, recipe, 0))
new_train_set=new_train_set+negative_train_set

for each in new_validation_set:
  x=random.choice(least_accessed)
  user=each[0]
  date=datetime.strftime(randomdate(),'%Y-%m-%d')
  while x in userCoocked[user]:
    x = random.choice(least_accessed)
  negative_validation_set.append((user,x,date,0))
  # negative_validation_set.append((user, recipe, 0))
new_validation_set=new_validation_set+negative_validation_set

#2 Predict on the validation set
def pop_pred(u,i,return1):
  if i in return1:
    return 1
  else:
    return 0
def pop_pred_on_val(return1):
  pred = []
  # Question 1: Use the vanilla predictor to predict whether someone has cooked something.
  for u,i,d in new_validation_set:
    this_result = pop_pred(u,i,return1)
    pred.append(this_result)
  pred_result=[x==y[2] for x,y in zip(pred,new_validation_set)]

  acc=sum(pred_result)/len(pred_result)
  return acc


def Jaccard(s1, s2):
  #Jaccard calculation function
  numer = len(s1.intersection(s2))
  if numer == 0:
    return 0
  denom = len(s1.union(s2))
  #No common between two sets, return 0
  if denom == 0:
    return 0
  # perform the intersection/union op.
  return numer / denom

def Cosine(s1, s2):
  # Not a proper implementation, operates on sets so correct for interactions only
  numer = len(s1.intersection(s2))
  denom = sqrt(len(s1)) * sqrt(len(s2))
  if denom == 0:
    return 0
  return numer / denom

SIM_METRIC = Jaccard
def check_recipe(u, g, SIM_METRIC):
  user_g = set(recipeCoocked[g]) #Users who cooked the test recipe
  set_each_g_=set()
  g_=userCoocked[u] #Recipe cooked by this user
  for each_g_ in g_:
    for this_u in recipeCoocked[each_g_]:
      set_each_g_.add(this_u)
  sim = SIM_METRIC(set_each_g_, user_g)
  return sim

def check_recipe_2(u, g, SIM_METRIC):
  item_u = set(userCoocked[u]) #Users who cooked the test recipe
  set_each_u_=set()
  u_=recipeCoocked[g] #Recipe cooked by this user
  for each_u_ in u_:
    for this_i in userCoocked[each_u_]:
      set_each_u_.add(this_i)
  sim = SIM_METRIC(set_each_u_, item_u)
  return sim
def sim_prepare(dataset, SIM_METRIC):
  print("Parsing Similarity ...")
  sim_dict_i=defaultdict(float)
  sim_dict_u = defaultdict(float)
  step = 20
  if (os.path.exists('sim_dict_i.txt') and os.path.exists('sim_dict_u.txt')):
    with open("sim_dict_i.txt", "rb") as fp:  # Unpickling
      sim_dict_i = pickle.load(fp)
    with open("sim_dict_u.txt", "rb") as fp:  # Unpickling
      sim_dict_u = pickle.load(fp)
  else:
    for step_i in range(step):
      start = int(step_i/step*len(dataset))
      end = int((step_i+1)/step*len(dataset))
      for u,i,_,_ in dataset[start:end]:
        sim_dict_i[(u,i)] = check_recipe(u, i, SIM_METRIC)
        sim_dict_u[(u,i)] = check_recipe_2(u, i, SIM_METRIC)
      progress = (step_i+1)*100/step
      print('Parsed %f%%' % progress)
    with open("sim_dict_i.txt", "wb") as fp:   #Pickling
      pickle.dump(sim_dict_i, fp)
    with open("sim_dict_u.txt", "wb") as fp:   #Pickling
      pickle.dump(sim_dict_u, fp)
  all_vals_i = [v for _, v in sim_dict_i.items() if v != 0]
  thre_i = np.percentile(all_vals_i,50)
  all_vals_u = [v for _, v in sim_dict_i.items() if v != 0]
  thre_u = np.percentile(all_vals_u,50)
  return sim_dict_i,thre_i,sim_dict_u,thre_u

sim_dict_i,thre_i,sim_dict_u,thre_u = sim_prepare(new_validation_set,SIM_METRIC)

def i_sim_high(u,i,SIM_METRIC):
  return check_recipe(u, i, SIM_METRIC)>thre_i

def u_sim_high(u,i,SIM_METRIC):
  return check_recipe_2(u, i, SIM_METRIC)>thre_u

def month_datapoints(percentage=20):
  gross_i = []
  gross_u = []
  for _,v in recipeDate.items():
    if len(v)>0:
      gross_i.append(median(v))

  month_i_thresh = np.percentile(gross_i, percentage)
  for _,v in userDate.items():
    if len(v)>0:
      gross_u.append(median(v))
  month_u_thresh = np.percentile(gross_i, percentage)
  return month_i_thresh,month_u_thresh

month_i_thresh,month_u_thresh = month_datapoints()

def i_month_near(i):
  if len(recipeDate[i])>0:
    return median(recipeDate[i])<=month_i_thresh
  else:
    return False

def u_month_near(u):
  if len(userDate[u])>0:
    return median(userDate[u])<=month_u_thresh
  else:
    return False # Nobody has ever done that before

def ui_features_construct(u,i,t):
  # return [1,pop_i(i),check_recipe(u, i, Cosine), in_return(i), parse_date(t)]
  # return [1, check_recipe(u, i, Cosine),check_recipe_2(u, i, Cosine), in_return(i),
  #         parse_date_u(u),parse_date_i(i)]
  return [1, in_return1(i), i_month_near(i), i_sim_high(u,i,SIM_METRIC)]
  # return [1,i_month_near(i), in_return1(i)]
def predict_features_construct(u,i):
  # return [1, check_recipe(u, i, Jaccard),check_recipe_2(u, i, Jaccard), in_return(i),
  #         parse_date_u(u),parse_date_i(i)]
  return [1, in_return1(i), i_month_near(i), i_sim_high(u,i,SIM_METRIC)]
  # return [1,i_month_near(i), in_return1(i)]
def parsing_dataset(dataset):
  # print("Parsing Start:")
  X=[]
  step = 20
  for i in range(step):
    start = int(i/step*len(dataset))
    end = int((i+1)/step*len(dataset))
    X+=[ui_features_construct(u,i,t) for u,i,t,_ in dataset[start:end]]
    # print("Parsed %f%%"%(i*100/step))
  return X
#1. Fitting the model

if (os.path.exists('parsed_trainset.txt')):
  with open("parsed_trainset.txt", "rb") as fp:  # Unpickling
    X = pickle.load(fp)
else:
  X = parsing_dataset(new_train_set)
  with open("parsed_trainset.txt", "wb") as fp:   #Pickling
    pickle.dump(X, fp)

y = [d for _,_,_,d in new_train_set]
import sklearn
mod = sklearn.linear_model.LogisticRegression()
#mod = sklearn.linear_model.SGDClassifier()

#2. Validate the model
if (os.path.exists('parsed_validset.txt')):
  with open("parsed_validset.txt", "rb") as fp:  # Unpickling
    X_valid = pickle.load(fp)
else:
  X_valid = parsing_dataset(new_validation_set)
  with open("parsed_validset.txt", "wb") as fp:   #Pickling
    pickle.dump(X_valid, fp)
label = [d for _,_,_,d in new_validation_set]
def parse_result(pred,y):
  TP = sum(np.logical_and(pred, y))
  FP = sum(np.logical_and(pred, np.logical_not(y)))
  TN = sum(np.logical_and(np.logical_not(pred), np.logical_not(y)))
  FN = sum(np.logical_and(np.logical_not(pred), y))
  return TP,FP,TN,FN

mod.fit(X,y)
min_acc = 0
mod.C = 0.01
for perc in np.arange(0.55,0.59,0.005):
  for perc_month in np.arange(10,20,5):
    return1 = pop_item_cut(perc)
    month_i_thresh,month_u_thresh = month_datapoints(perc_month)
    X = parsing_dataset(new_train_set)
    X_valid = parsing_dataset(new_validation_set)
    mod.fit(X,y)
    pred_valid = mod.predict(X_valid)
    TP,FP,TN,FN = parse_result(pred_valid,label)
    acc = (TP+TN)/(TP+TN+FP+FN)
    FNR = FN/(FN+TP)
    FPR = FP/(FP+TN)
    BER=0.5*(FNR+FPR)
    print("perc:%f, perc_month:%f,acc:%f, ber:%f, fnr:%f, fpr:%f" % (perc, perc_month, acc, BER, FNR, FPR))
    if acc>min_acc:
      best_perc = perc
      best_perc_month = perc_month
      min_acc = acc

return1 = pop_item_cut(best_perc)
month_i_thresh, month_u_thresh = month_datapoints(best_perc_month)
X = parsing_dataset(new_train_set)
mod.fit(X,y)

pos_count=0
neg_count=0
predictions = open("predictions_Made.txt", 'w')

for l in open("stub_Made.txt"):
  if l.startswith("user_id"):
    #header
    predictions.write(l)
    continue
  u,i = l.strip().split('-')

  feature = predict_features_construct(u,i)
  feature = np.array(feature).reshape([1, -1])
  result_ui = mod.predict(feature)[0]
  if result_ui==1:
    pos_count+=1
  else:
    neg_count+=1
  result_ui = str(result_ui)
  predictions.write(u + '-' + i + ","+result_ui+"\n")

predictions.close()
print(pos_count/(pos_count+neg_count))

