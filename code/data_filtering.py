import numpy as np
import json
import argparse
import os
from nltk import PorterStemmer
import ast 
import editdistance
import tqdm
parser = argparse.ArgumentParser(description='Data Handler for Covid wikification')
parser.add_argument('--entities',required= True, type=str, nargs=1,
                    help='Path to entities')
parser.add_argument('--training',required= True, type=str, nargs=1,
                    help='Path to training')
parser.add_argument('--misc',required= True, type=str, nargs=1,
                    help='Path to rest')
parser.add_argument('--lim',required= True, type=int, nargs=1,
                    help='Path to rest')
parser.add_argument('--out',required= True, type=str, nargs=1,
                    help='Path to rest')
parser.add_argument('--pickled',required= False, default = "False",type=str, nargs=1,
                    help='use pickled')

args = parser.parse_args()
np.random.seed(42)

limit= args.lim[0]

data_fraction = 1.0

train_fraction = 0.6
dev_fraction = 0.2
test_fraction = 0.2

print("Reading Data...")

f = open(args.entities[0],'rb')
entities = json.load(f)
f.close()

import pickle

if  args.pickled[0] == "False":
    print("Loading from File")
    f = open(args.training[0],'rb')
    line =  f.readline()
    data = []
    i = 0
    while line:
        sentence = json.loads(line)
        data.append(sentence)
        line =  f.readline()
        i+=1
        if i == limit:
            break       
    f.close()
    data = np.array(data)
else:
    print("reading pickled...")
    f = open("../../data/pickled_training_data.pkl",'rb')
    data = pickle.load(f)
    f.close()
############################################3
import pandas as pd
a = pd.read_csv(os.path.join(args.misc[0],"medical_terms.txt"), sep = ' ') #List of medical terms
keep =["General","t"]
a = a[keep]
a = a.fillna("()")
monograms = a["General"].values
med_words = list(monograms.copy().reshape(-1))
stemmer = PorterStemmer()
monogram_stems = []
for m in monograms: #Stems words
    stem = stemmer.stem(m)
    monogram_stems.append(stem)

monogram_stems = np.array(monogram_stems)

f = open(os.path.join(args.misc[0],"entities_to_parents.tuple"),'r',encoding = "utf8")
content = f.readlines()
word_types = {}
for c in content:
    if "null" not in c:
        t = ast.literal_eval(c)
        word_types[t[1].lower()] = t[2]

###################################################################

f = open(os.path.join(args.misc[0],"cities_countries.pkl"),"rb")
cities,countries = pickle.load(f)
f.close()

f = open(os.path.join(args.misc[0],"keep_type_corrected.pkl"),"rb")
keep_type = pickle.load(f)
f.close()






print("Finished Reading.")

print("Finding Relevant Concepts")

relevant_concepts = {}

for c in entities:
    relevant_concepts[c.lower()] = True

for concept in relevant_concepts.keys():
    keep_concept = True
    is_place = (concept in cities) or (concept in countries) or "country" in concept.lower() or "city" in concept.lower()
    if is_place:
        relevant_concepts[concept] = False
        continue
    
    is_med = stemmer.stem(concept) in monogram_stems


banned_types = ["tourist attraction","republic","island","literary work","neighborhood","country","city","human","town","place","family","continent","area","language","relig","empire","colonial","monarchy","sovereign state","island nation", "democratic republic","overseas territory","historical country"]

for concept in relevant_concepts.keys():
    if relevant_concepts[concept]:
        if concept in word_types.keys():
            types = word_types[concept]
            keep = any([keep_type[t] for t in types])
            if not keep:
                relevant_concepts[concept] = False

        else:
            pass

n_initial_concepts = np.sum(np.array(list(relevant_concepts.values()),np.int))
print("Found initial interesting concepts:", n_initial_concepts,"from",len(relevant_concepts))

#################################################################################
print("Filtering sentences")

concept_counts = {}
for concept in relevant_concepts.keys():
    if relevant_concepts[concept]:
        concept_counts[concept] = 0

def process_sentence(s, relevant_concepts):
    flag = False
    concepts= []
    for i in range(len(s)):
        
        c = s[i][1].split("_")[-1]

        if c in relevant_concepts:
            if not relevant_concepts[c]:
                
                s[i][1] = "O"
                
                flag = True
            elif relevant_concepts[c]:
                concepts.append(c)
        else:
            s[i][1] = "O"
    
    concepts = np.unique(concepts)
    
    
        
    return s,flag, concepts

def check_concept_counts(concepts,thresh = 50):
    for c in concepts:
        if concept_counts[c]>=thresh:
            return False
    
    return True
print("------------------------------------------")
filtered_data = []

for d in tqdm.tqdm(data):

    d_,_,concepts = process_sentence(d,relevant_concepts)
    #print(concepts)
    if check_concept_counts(concepts):
        
        for c in concepts:
            concept_counts[c]+=1
        if len(concepts)>0:
            filtered_data.append(d_)

print("Pre-filtering", len(data), "Post filtering",len(filtered_data))
del data

for concept in concept_counts:
    if concept_counts[concept]<5:
        relevant_concepts[concept] = False

data = filtered_data
filtered_data = []

for d in tqdm.tqdm(data):
    d_,_,concepts = process_sentence(d, relevant_concepts)
    
    if len(concepts)>0:
        filtered_data.append(d_)

print("Pre-filtering", len(data), "Post filtering",len(filtered_data))

filtered_data = np.array(filtered_data)
np.random.shuffle(filtered_data)

size = len(filtered_data)
inds = np.array([0,size*train_fraction,size*train_fraction+size*dev_fraction,size],np.int)

train_set = filtered_data[:inds[1]]
dev_set = filtered_data[inds[1]:inds[2]]
test_set = filtered_data[inds[2]:]
full_labels = list(concept_counts.keys())

#########################################################################################    


frqs = {k: v for k, v in sorted(concept_counts.items(), key=lambda item: int(item[1]))}

#########################################################################################
print("Creating ambiguous dataset")

token_label_counts = {}

for d in filtered_data:
    for a,b in d:
        if b!="O":
            if a not in token_label_counts.keys():
                token_label_counts[a] = {}
            if b not in token_label_counts[a].keys():
                token_label_counts[a][b] =1
            else:
                token_label_counts[a][b]+=1

token_uncertain= {token:True for token in token_label_counts.keys()}
certain_tokens = {}
for token, labels in token_label_counts.items():
    
    total = np.sum(list(labels.values()))
    m=  np.max(list(labels.values()))
    ind  = np.argmax(list(labels.values()))
    l = max(labels,key = labels.get)
    if m/total >0.9:
        token_uncertain[token] = False
        certain_tokens[token] = l


def is_ambiguous(s):
    amb = False
    for a,b in s:
        if b!="O":
            if token_uncertain[a]:
                amb = True
                break
    #if not amb:
    #    print(s)
        
    return amb

def get_relevant_concepts(s, relevant):
    concepts = []
    for a,b in s:
        c = b.split("_")[-1]
        if c in relevant:
            if relevant[c]:
                concepts.append(c)
        
    concepts = np.unique(concepts)

    return concepts

ambiguous_relevant_concepts = {c:f for c,f in relevant_concepts.items()}
ambiguous_counts = {}
ambiguous_data = []
for d in filtered_data:
     if is_ambiguous(d):
         ambiguous_data.append(d)
         concepts = get_relevant_concepts(d,relevant_concepts)
         for c in concepts:
            if c not in ambiguous_counts:
                ambiguous_counts[c]=1
            else:
                ambiguous_counts[c]+=1


for concept in ambiguous_counts:
    if ambiguous_counts[concept]<5:
        ambiguous_relevant_concepts[concept] = False

_data = ambiguous_data
ambiguous_data = []
for d in tqdm.tqdm(_data):
    d_,_,concepts = process_sentence(d, ambiguous_relevant_concepts)
    if len(concepts)>0:
        ambiguous_data.append(d_)

ambiguous_counts = {}
for d in ambiguous_data:
         concepts = get_relevant_concepts(d,ambiguous_relevant_concepts)
         for c in concepts:
            if c not in ambiguous_counts:
                ambiguous_counts[c]=1
            else:
                ambiguous_counts[c]+=1

print("All", len(filtered_data), "Ambiguous",len(ambiguous_data))

a_frqs = {k: v for k, v in sorted(ambiguous_counts.items(), key=lambda item: int(item[1]),reverse=True)}
frqs = {k: v for k, v in sorted(concept_counts.items(), key=lambda item: int(item[1]),reverse=True)}
print("Concepts in ambiguous dataset", len(frqs))

a_size = len(ambiguous_data)
inds = np.array([0,a_size*train_fraction,a_size*train_fraction+a_size*dev_fraction,a_size],np.int)

a_train_set = ambiguous_data[:inds[1]]
a_dev_set = ambiguous_data[inds[1]:inds[2]]
a_test_set = ambiguous_data[inds[2]:]
a_labels = list(ambiguous_counts.keys())


def output_list(train,fname, label_list = ['O']):
    labels = []
    train_file = open(os.path.join(args.out[0],fname),'w', encoding="utf8")
    for s in train:
        sentence_str = ""
        
        for a,b in s:
            b_ = str(b).replace(" ","_")
            sentence_str += str(a+" "+b_+"\n")
        
        train_file.write(sentence_str+"\n")

            

    train_file.close()
    return 

def output_ljson(data,fname):
    #print()
    train_file = open(os.path.join(args.out[0],fname),'w', encoding="utf8")
    for s in data:
        train_file.write(str(s)+"\n")
    train_file.close()



output_list(a_train_set,"ambiguous_data/train.txt", label_list = a_labels)
output_list(a_dev_set,"ambiguous_data/dev.txt", label_list = a_labels)
output_list(a_test_set,"ambiguous_data/test.txt", label_list = a_labels)

output_ljson(a_train_set,"ambiguous_data/train.ljson")
output_ljson(a_dev_set,"ambiguous_data/dev.ljson")
output_ljson(a_test_set,"ambiguous_data/test.ljson")

output_list(train_set,"full_data/train.txt", label_list = a_labels)
output_list(dev_set,"full_data/dev.txt", label_list = a_labels)
output_list(test_set,"full_data/test.txt", label_list = a_labels)

output_ljson(train_set,"full_data/train.ljson")
output_ljson(dev_set,"full_data/dev.ljson")
output_ljson(test_set,"full_data/test.ljson")


label_file = open(os.path.join(args.out[0],"ambiguous_data/labels.txt"),'w', encoding = 'utf8')
for l in a_labels:
    label_file.write("wiki_"+str(l).replace(" ","_")+"\n")
label_file.close()


label_file = open(os.path.join(args.out[0],"full_data/labels.txt"),'w', encoding = 'utf8')
for l in full_labels:
    label_file.write("wiki_"+str(l).replace(" ","_")+"\n")
label_file.close()


f = open(os.path.join(args.out[0],"full_data/freqs.txt"),'w',encoding = "utf8")
tmp_ = [str(f)+","+str(l)+"\r\n" for f,l in frqs.items()]
f.writelines(tmp_)
f.close()


f = open(os.path.join(args.out[0],"ambiguous_data/freqs.txt"),'w',encoding = "utf8")
tmp_ = [str(f)+","+str(l)+"\r\n" for f,l in a_frqs.items()]
f.writelines(tmp_)
f.close()


data_info = "Full dataset:\r\n{s1} sentences\r\n{n_conc} concepts\r\nAmbiguous dataset: \r\n{s2} sentences\r\n{n_aconc} concepts".format(s1 = size,n_conc = len(full_labels), s2 = a_size, n_aconc = len(a_labels))

f = open(os.path.join(args.out[0],"info.txt"),'w',encoding = "utf8")
f.write(data_info)
f.close()
