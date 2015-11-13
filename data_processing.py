import json
import collections
import random

user_filename = 'yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_user.json'
review_filename = 'yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json'
steps = 7
k_users = 300 
train_size_rate = 0.7
########################################################
def multiLinesLoad(json_file_name):
	file = open(json_file_name, 'r')
	j = []
	while(1):
		line = file.readline()
		if not line:
			break
		j.append(json.loads(line))
	return j

def multiLinesWrite(unicodeList, file_name):
	f = open(file_name, 'w')
	j_dumped = []
	for item in unicodeList:
		j_dumped.append(json.dumps(item)+'\n')
	f.writelines(j_dumped)
	f.close()
########################################################

print "="*30
print "Step 1/%d: Opening user json file %r..." % (steps, user_filename),
f = open(user_filename, 'r')
print "Done!"

print "="*30
print "Step 2/%d: Creating json object from %r..." % (steps, user_filename),
j = []
while(1):
# for i in range(10):
	line = f.readline()
	if not line:
		break
	j.append(json.loads(line))
print "Done!"

print "="*30
print "Step 3/%d: Sorting users by review counts, and select users..." % steps
j.sort(key = lambda x: x['review_count'])
j.reverse()
total_reviews = 0
# user_id_list = []
user_id_counter = collections.Counter()
for i in range(k_users):
	total_reviews += j[i]['review_count']
	# user_id_list.append(j[1000*i+1000]['user_id'])
	user_id_counter[j[i]['user_id']] = 0
print "Total reviews of the selected %d users = %d" % (k_users, total_reviews)
print "Done!"
f.close()

print "="*30
print "Step 4/%d: Opening review json file %r..." % (steps, review_filename),
f = open(review_filename, 'r')
print "Done!"

j_selected_reviews = []
print "="*30
print "Step 5/%d: Creating json object from %r..." % (steps, review_filename)
review_count = 0
while(1):
	line = f.readline()
	if not line:
		break
	temp = json.loads(line)
	if temp['user_id'] in user_id_counter:
		j_selected_reviews.append(temp)
		user_id_counter[temp['user_id']] += 1
		review_count += 1
print "User_id_counter: %r" % user_id_counter
print "%d reviews are drawn." % review_count
print "Done!"
f.close()


print "="*30
print "Step 6/%d: Splitting training and testing set..." % steps
random.shuffle(j_selected_reviews)
train_size = int(train_size_rate * review_count)
print "Training set size: %r" % train_size
test_size = review_count - train_size
print "Testing set size: %r" % test_size
j_train = []
j_test = []
for item in j_selected_reviews:
	if len(j_train) >= train_size:
		j_test.append(item)
		continue
	else:
		j_train.append(item)
		continue
	toss = random.random()
	if toss < train_size_rate:
		j_train.append(item)
	else:
		j_test.append(item)
if len(j_train) == train_size and len(j_test) == test_size:
	print "Size check passed."
else:
	print "Error: Size check failed!!!!"
print "Done!"

print "="*30
print "Step 7/%d: Saving training and testing json files..." % steps 
multiLinesWrite(j_train, 'train.json')
multiLinesWrite(j_test, 'test.json')
print "Done!"







