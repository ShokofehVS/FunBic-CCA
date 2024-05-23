import urllib.request
import numpy as np

# get data
# url = "http://arep.med.harvard.edu/biclustering/lymphoma.matrix"
# lines = urllib.request.urlopen(url).read().strip().split('\n')
# # insert a space before all negative signs
# lines = list(' -'.join(line.split('-')).split(' ') for line in lines)
# lines = list(list(int(i) for i in line if i) for line in lines)
# data = np.array(lines)
# print(data)

from urllib.request import urlopen
#check text for curse words
def check_profanity():
    with urlopen("http://arep.med.harvard.edu/biclustering/lymphoma.matrix") as f:
        lines = f.read().decode('utf-8').strip().split('\n')
        lines = list(' -'.join(line.split('-')).split(' ') for line in lines)

        lines = list(list(int(i) for i in line if i) for line in lines)
        data = np.array(lines)
        print(data, data.shape)
        if lines:
            if "true" in lines:
                print("There is a profane word in the document")


check_profanity()