import json
import csv
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import math
import re
import pickle as p
import time
import collections
from collections import defaultdict

stats_export_path = 'C:/games/Kaggle/Digits/'

train_file_name = 'train.csv'
test_file_name = 'test.csv'

im_len = 28
darkness_cutoff = 10

'''
Directions maps:
701
6.2
543
'''
dirtable=[[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1]]

# Set to TRUE to ignore single-length contours during contouring
contouring_ignore_artifacts = True

dubiousness_coeff = 1

certainty_cutoff = 0.5


# LOAD TRAIN DATA    
'''
f = open('C:/games/Kaggle/Digits/truncated.csv')
data_train = csv.load(f)
f.close()

print(data_train)
'''

# If ones_or_zeros is 0, checks if the image is all 1's.
# IF ones_or_zeros is 1, checks if the image is all 0's.
def check_if_image_is_filled(image, ones_or_zeros):
    return_value = True
    for i in range(0,im_len):
        for j in range(0,im_len):
            if image[i][j] == ones_or_zeros:
                return_value = False
                break
                
    return return_value
          
def recursive_fill(image, i, j):
    
    if image[i][j] == 0:
            image[i][j] = 1
    
    if i>0 and j>0:
        if image[i-1][j-1] == 0:
            image[i-1][j-1] = 1
            recursive_fill(image,i-1,j-1)

    if i>0:
        if image[i-1][j] == 0:
            image[i-1][j] = 1
            recursive_fill(image,i-1,j)
    
    if i>0 and (j+1) < im_len:
        if image[i-1][j+1] == 0:
            image[i-1][j+1] = 1
            recursive_fill(image,i-1,j+1)
            
    if j>0:
        if image[i][j-1] == 0:
            image[i][j-1] = 1
            recursive_fill(image,i,j-1)

    if (j+1) < im_len:
        if image[i][j+1] == 0:
            image[i][j+1] = 1
            recursive_fill(image,i,j+1)
            
    if (i+1) < im_len and j>0:
        if image[i+1][j-1] == 0:
            image[i+1][j-1] = 1
            recursive_fill(image,i+1,j-1)

    if (i+1) < im_len:
        if image[i+1][j] == 0:
            image[i+1][j] = 1
            recursive_fill(image,i+1,j)
    
    if (i+1) < im_len and (j+1) < im_len:
        if image[i+1][j+1] == 0:
            image[i+1][j+1] = 1
            recursive_fill(image,i+1,j+1)
            
#
def fill_image(image):
    # Find starting point and call the recursive fill
    #print("CALLED")
    breakout = False
    for i in range(0,im_len):
        if breakout:
            break
        for j in range(0,im_len):   
            if breakout:
                break
            if image[i][j] == 0:
                recursive_fill(image,i,j)
                #print_image2(image)
                breakout = True
            
def image_count_holes(image):
    copy = np.copy(image)
    counter = 0
    while not check_if_image_is_filled(copy,0):
        fill_image(copy)
        counter+=1
        
    return counter

def horizontal_intersect_count(image):
    counts = np.zeros(im_len)

    for i in range(0,im_len):
        current = image[0][0]
        for j in range(1,im_len):
            nextone = image[i][j]
            if nextone != current and current == 0:
                counts[i]+=1
            current = nextone
            
    return int(np.max(counts))

def vertical_intersect_count(image):
    counts = np.zeros(im_len)

    for i in range(0,im_len):
        current = image[0][0]
        for j in range(1,im_len):
            nextone = image[j][i]
            if nextone != current and current == 0:
                counts[i]+=1
            current = nextone
            
    return int(np.max(counts))

def find_center_of_mass(image):
    count = 0
    sumx = 0
    sumy = 0
    
    for i in range(0,im_len):
        for j in range(0,im_len):
            if image[i][j]>0:
                count+=1
                sumx+=i
                sumy+=j
    return [count, sumx/float(count), sumy/float(count)]

def check_if_pixel_is_internal(image,i,j):
    return_value = True
    
    counter = 0
    corner = False    
    
    if i>0 and j>0:
        if image[i-1][j-1] == 0:
            return_value = False
            counter +=1
            corner = True
            
    if i>0:
        if image[i-1][j] == 0:
            return_value = False
            counter +=1

    if i>0 and (j+1)<im_len:
        if image[i-1][j+1] == 0:
            return_value = False
            counter +=1
            corner = True
            
    if j>0:
        if image[i][j-1] == 0:
            return_value = False
            counter +=1
            
    if (j+1)<im_len:
        if image[i][j+1] == 0:
            return_value = False
            counter +=1
            
    if (i+1) < im_len and j>0:
        if image[i+1][j-1] == 0:
            return_value = False
            counter +=1
            corner = True
    
    if (i+1) < im_len:
        if image[i+1][j] == 0:
            return_value = False
            counter +=1
            
    if (i+1) < im_len and (j+1) < im_len:
        if image[i+1][j+1] == 0:
            return_value = False
            counter +=1
            corner = True
    
    if counter == 1 and corner:
        return_value = True
    
    return return_value            
            
def excavate_image(image):
    # Create empty array to store values we want to delete
    to_delete = np.zeros((im_len,im_len))
    for i in range(0,im_len):
        for j in range(1,im_len):
            if image[i][j] == 1 and check_if_pixel_is_internal(image,i,j):
                to_delete[i][j] = 1
                
    # Delete marked pixels
    for i in range(0,im_len):
        for j in range(1,im_len):
            if to_delete[i][j] == 1:
                image[i][j] = 0
                
    #print("EXCAVATED:")
    #print_image2(image)


def image_find_filled_pixel(image):
    for i in range(0,im_len):
        for j in range(0,im_len):
            if image[i][j] == 1:
                return [i,j]

    return [-1,-1]
      
def check_if_there_is_filled_pixel_in_dir(image, i,j,givendir):
    if givendir == 0 and i>0 and image[i-1][j]==1:
        return True
        
    if givendir == 1 and i>0 and (j+1)<im_len and image[i-1][j+1]==1:
        return True
        
    if givendir == 2 and (j+1)<im_len and image[i][j+1]==1:
        return True
        
    if givendir == 3 and (i+1)<im_len and (j+1)<im_len and image[i+1][j+1]==1:
        return True
        
    if givendir == 4 and (i+1)<im_len and image[i+1][j]==1:
        return True
        
    if givendir == 5 and (i+1)<im_len and j>0 and image[i+1][j-1]==1:
        return True
        
    if givendir == 6 and j>0 and image[i][j-1]==1:
        return True
        
    if givendir == 7 and i>0 and j>0 and image[i-1][j-1]==1:
        return True
        
    return False
    
'''
Directions maps:
701
6.2
543

We try to move clockwise.
We return [x,y,dir] in case of success, where (x,y) are the coordinates of
the next pixel, and dir is the direction of the next pixel.
We return [-1,-1,-1] in case of failure.
'''
def contouring_take_step(image,i,j,lastdir):
    # Delete current pixel from image
    image[i][j] = 0
    
    # Search for next pixel
    for q in range(0,7):
        dircheck = (lastdir-3+q)%8
        if check_if_there_is_filled_pixel_in_dir(image, i, j, dircheck):
            return [i+dirtable[dircheck][0], j+dirtable[dircheck][1], dircheck]
            
    return [-1,-1,-1]
            

def trace_contours(image):
    # Excavate to leave only contours...
    excavate_image(image)

    # Create holding list
    contours = []
    
    #counter = 0    
    
    while not check_if_image_is_filled(image, 1):
        
        '''
        counter+=1        
        
        if counter > 7:
            print("PROBLEM!")
            break
        '''        
        
        start = image_find_filled_pixel(image)
        
        lastdir = 0
        if not check_if_there_is_filled_pixel_in_dir(image,start[0],start[1],4):
            lastdir = 1
        
        dir_str = str(lastdir)
        
        step_details = contouring_take_step(image, start[0],\
            start[1], lastdir)
        while(lastdir != -1):
            lastdir = step_details[2]
            dir_str+=str(lastdir)
            step_details = contouring_take_step(image, step_details[0],\
                step_details[1], lastdir)
            
        if not (contouring_ignore_artifacts and len(dir_str)<=3):
            contours.append(dir_str)
        
        #print("CONTOUR TRACED")
        #print_image2(image)
        
    return contours
        

def contours_maxlength_count(contours):
    lengths = []
    for i in range(0, len(contours)):
        lengths.append(len(contours[i]))
        
    return [max(lengths), len(contours)]

def contours_straightness_tendency(contours):
    coeffs = []
    sumlen = 0
    for i in range(0, len(contours)):
        cursum = 0
        strng = contours[i]       
        
        cur = strng[0]
        for j in range(1, len(contours[i])):
            nxt = strng[j]
            if nxt != "-":
                cursum+=((int(nxt)-int(cur))%8)
            else:
                break
            cur = nxt
        coeffs.append(cursum)
        sumlen+= len(strng)
        
        return sum(coeffs)/float(sumlen)

def extract_image_statistics(data, index):
    # Process image data into a double array. 

    # change this to one if processing train set
    # add bool to avoid problem
    addone = 0
   
    image = np.zeros((im_len, im_len))
    for i in range(0,im_len):
        for j in range(0,im_len):
            image[i][j] = int(int(data[index][addone+i*im_len+j])>darkness_cutoff)
    
    holes = image_count_holes(image)
    #print("HOLES:")
    #print(holes)
    
    horizontal_cuts = horizontal_intersect_count(image)
    #print("MAX HORIZONTAL INTERSECTIONS:")
    #print(horizontal_cuts)
    
    vertical_cuts = vertical_intersect_count(image)
    #print("MAX VERTICAl INTERSECTIONS:")
    #print(vertical_cuts)
    
    mass_list = find_center_of_mass(image)
    #print("PIXEL COUNT:")
    #print(mass_list[0])
    #print("CENTER OF MASS x:")
    #print(mass_list[1])
    #print("CENTER OF MASS y:")
    #print(mass_list[2])

    contours = trace_contours(np.copy(image))
    #print("CONTOURS:")
    #print(contours)
    
    contours_max_cnt = contours_maxlength_count(contours)
    #print("LONGEST CONTOUR:")
    #print(contours_max_cnt[0])
    #print("COUNT OF CONTOURS:")
    #print(contours_max_cnt[1])
    straightness_tendency = contours_straightness_tendency(contours)

    return [holes, horizontal_cuts, vertical_cuts, mass_list[0], mass_list[1],\
        mass_list[2], contours_max_cnt[0], contours_max_cnt[1],\
        straightness_tendency]

def print_image2(image):  
    for i in range(0,im_len):
        strng=""
        for j in range(0,im_len):
            if i==0 or i==im_len-1 or j==0 or j==im_len-1:
                strng+="*"
            elif image[i][j] > 0:
                strng+="x"
            else:
                strng+=" "
                
        print(strng)

def print_image(data, index):
    for i in range(0,im_len):
        strng=""
        for j in range(0,im_len):
            if i==0 or i==im_len-1 or j==0 or j==im_len-1:
                strng+="*"
            elif int(data[index][1+i*im_len+j]) > darkness_cutoff:
                strng+="x"
            else:
                strng+=" "
                
        print(strng)


def extract_and_save_statistics(data, filename, alsogetnumbers):

    data_len = len(data)
    
    t0 = time.clock()    
    
    # Define lists for storage    
    numbers = [] # not used for Test data    
    holes = []
    horizontal_cuts = []
    vertical_cuts = []
    pixel_count = []
    mass_x = []
    mass_y = []
    longest_contour = []
    contours_count = []
    straightness_coeff = []
    
    for i in range(1, data_len):
        if alsogetnumbers:
            numbers.append(int(data[i][0]))
        image_stats = extract_image_statistics(data, i)
        holes.append(image_stats[0])
        horizontal_cuts.append(image_stats[1])
        vertical_cuts.append(image_stats[2])
        pixel_count.append(image_stats[3])
        mass_x.append(image_stats[4])
        mass_y.append(image_stats[5])
        longest_contour.append(image_stats[6])
        contours_count.append(image_stats[7])
        straightness_coeff.append(image_stats[8])
        
        print(i)
        
        
    indexset = range(0,data_len-1)
    
    # Save various pieces of data in a csv file for purposes of analysis.
    if alsogetnumbers:
        log_answers = [indexset,numbers, holes, horizontal_cuts, vertical_cuts,\
            pixel_count, mass_x, mass_y, longest_contour, contours_count,\
            straightness_coeff]
    else:
        log_answers = [indexset,holes, horizontal_cuts, vertical_cuts,\
            pixel_count, mass_x, mass_y, longest_contour, contours_count,\
            straightness_coeff]
    with open(stats_export_path+filename+'.csv', 'wb') as test_file:
        file_writer = csv.writer(test_file)
        
        if alsogetnumbers:
            header_string = ["index", "numbers", "holes", "horizontal_cuts",\
            "vertical_cuts","pixel_count", "mass_x", "mass_y",\
            "longest_contour", "contours_count", "straightness"]
        else:
            header_string = ["index", "holes", "horizontal_cuts",\
            "vertical_cuts","pixel_count", "mass_x", "mass_y",\
            "longest_contour", "contours_count", "straightness"]
        
        file_writer.writerow(header_string)
        for i in range(data_len-1):
            file_writer.writerow([x[i] for x in log_answers])
        
    print("EXAMINED THE FOLLOWING NUMBER OF IMAGES:")
    print(data_len)
    print("WROTE IMAGE STATISTICS TO FILE. TIME TAKEN FOR ANALYSIS:")
    
    print(time.clock() - t0)

# Process Train set statistics, collected from file "filename".
def process_statistics(stats):
    
    '''
    print(len(stats["numbers"][0]))
    holes_1 = [holes for (digit,holes) in zip(stats["numbers"],stats["holes"]) \
        if (digit == "3")]    
    freqs_holes_1 = collections.Counter(holes_1)
    #print(holes_1)
    print(freqs_holes_1)
    '''
    
    # Generate frequency tables and averages
    
    # HOLES    
    holes_freqs = []
    for i in range(0,10):
        current_list = [int(holes) for (digit,holes) in zip(stats["numbers"],\
            stats["holes"]) \
            if (digit == str(i))]    
        freqs_table = collections.Counter(current_list)
        holes_freqs.append(freqs_table)
        
    # HORIZONTAL CUTS    
    hor_cuts_freqs = []
    for i in range(0,10):
        current_list = [int(cuts) for (digit,cuts) in zip(stats["numbers"],\
            stats["horizontal_cuts"]) \
            if (digit == str(i))]    
        freqs_table = collections.Counter(current_list)
        hor_cuts_freqs.append(freqs_table)

     # VERTICAL CUTS
    ver_cuts_freqs = []
    for i in range(0,10):
        current_list = [int(cuts) for (digit,cuts) in zip(stats["numbers"],\
            stats["vertical_cuts"]) \
            if (digit == str(i))]    
        freqs_table = collections.Counter(current_list)
        ver_cuts_freqs.append(freqs_table) 
        
    # PIXEL COUNT
    pixel_count_avrgs = []
    for i in range(0,10):
        current_list = [int(pxl) for (digit,pxl) in zip(stats["numbers"],\
            stats["pixel_count"]) \
            if (digit == str(i))]
        pixel_count_avrgs.append(sum(current_list)/float(len(current_list)))
        
    # MASS X
    massx_avrgs = []
    for i in range(0,10):
        current_list = [float(massx) for (digit,massx) in zip(stats["numbers"],\
            stats["mass_x"]) \
            if (digit == str(i))]
        massx_avrgs.append(sum(current_list)/float(len(current_list)))
        
    # MASS Y
    massy_avrgs = []
    for i in range(0,10):
        current_list = [float(massy) for (digit,massy) in zip(stats["numbers"],\
            stats["mass_y"]) \
            if (digit == str(i))]
        massy_avrgs.append(sum(current_list)/float(len(current_list)))
        
    # CONTOUR LENGTH
    cont_len_avrgs = []
    for i in range(0,10):
        current_list = [int(cnt) for (digit,cnt) in zip(stats["numbers"],\
            stats["longest_contour"]) \
            if (digit == str(i))]
        cont_len_avrgs.append(sum(current_list)/float(len(current_list)))
    
    return [holes_freqs, hor_cuts_freqs, ver_cuts_freqs, pixel_count_avrgs,\
        massx_avrgs, massy_avrgs, cont_len_avrgs]
    #print(pixel_count_avrgs)
    
    #print(holes_freqs)
    
def get_score_from_freq_table(freq_table, value):
    maxval = max(freq_table.values())    
        
    if value<1:
        return 0
    if value>maxval:
        return 0
        
    returnval = freq_table[value]/float(sum(freq_table.values()))    
    return returnval

def get_scores_from_freq_table(freq_tables, value):
    scores = []
    
    for i in range(0,10):
        scores.append(get_score_from_freq_table(freq_tables[i], value))
        
    return scores

def get_scores_from_avrg_table(avrg_table, value):
    # This should not be recalculated every time...    
    #tbl_range = max(avrg_table) - min(avrg_table)
    
    '''
    scores = []
    tmp = []
    for i in range(0,10):
        diff = abs(value-avrg_table[i])        
        tmp.append(diff)
        
    for i in range(0,10):
        maxdiff = max(tmp)
        scores.append(tmp[i]/maxdiff)
    '''
    
    scores = []    
    tbl_range = max(avrg_table) - min(avrg_table)
    for i in range(0,10):
        diff = abs(value-avrg_table[i])
        
        if tbl_range>diff:
            scores.append((tbl_range-diff)/tbl_range)
            #print("success")
        else:
            scores.append(0)
            #print("fail")
    
    return scores

# Generates predictions for the numbers whose processed data can be found
# in STATS, given the Train-set statistics passed in AVERAGES.
def generate_predictions(averages, stats):
    
    scores = []
    
    data_len = len(stats["holes"])
    
    #print(data_len)
    #print(len(averages))
    #print(len(stats))    
    
    #data_len = 100
    
    holes_weight = 2
    hor_cuts_weight = 1
    ver_cuts_weight = 1
    pxl_cnt_weight = 1
    massx_weight = 0
    massy_weight = 2*massx_weight
    contour_weight = 1
    
    
    for i in range(0,data_len):
        points = {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0, 7:0.0,\
            8:0.0, 9:0.0}
        
        # HOLES
        holes_score = get_scores_from_freq_table(averages[0],\
            int(stats["holes"][i]))
        for j in range(0,10):
            points[j]+=holes_weight*holes_score[j]
            
        # HORIZONTAL CUTS
        hor_cuts_score = get_scores_from_freq_table(averages[1],\
            int(stats["horizontal_cuts"][i]))
        for j in range(0,10):
            points[j]+=hor_cuts_weight*hor_cuts_score[j]

        # VERTICAL CUTS
        ver_cuts_score = get_scores_from_freq_table(averages[2],\
            int(stats["vertical_cuts"][i]))
        for j in range(0,10):
            points[j]+=ver_cuts_weight*ver_cuts_score[j]
            
        # PIXEL COUNTS
        pxl_cnt_score = get_scores_from_avrg_table(averages[3],\
            int(stats["pixel_count"][i]))
        for j in range(0,10):
            points[j]+=pxl_cnt_weight*pxl_cnt_score[j]
            
        # MASS X
        massx_score = get_scores_from_avrg_table(averages[4],\
            float(stats["mass_x"][i]))
        for j in range(0,10):
            points[j]+=massx_weight*massx_score[j]
            
        # MASS X
        massy_score = get_scores_from_avrg_table(averages[5],\
            float(stats["mass_y"][i]))
        for j in range(0,10):
            points[j]+=massy_weight*massy_score[j]
            
        # CONTOUR LENGTH
        cont_score = get_scores_from_avrg_table(averages[6],\
            float(stats["longest_contour"][i]))
        for j in range(0,10):
            points[j]+=contour_weight*cont_score[j]
            
        scores.append(points)
        #print(points)

    predictions = []
    for i in range(0,data_len):
        points = scores[i]
        
        sorting = sorted(points, key=points.get, reverse=True)
        
        predictions.append(sorting[0])
        
    indexset = range(1,data_len+1)
    
    # Save various pieces of data in a csv file for purposes of analysis.
    log_answers = [indexset, predictions]

    with open(stats_export_path+"predictions"+'.csv', 'wb') as test_file:
        file_writer = csv.writer(test_file)
        
        header_string = ["ImageId", "Label"]

        file_writer.writerow(header_string)
        for i in range(data_len):
            file_writer.writerow([x[i] for x in log_answers])        
        
        
def score_train_predictions(predictions, stats):
    counter1 = 0
    data_len = len(stats)
    for i in range(0,data_len):
        #print("ACTUAL: "+ stats["numbers"][i] + "; PREDICTED: "+\
        #    str(predictions[i])+"; DUBIOUS? "+dubious[i])
        if int(stats["numbers"][i]) != predictions[i]:
            counter1+=1
  
    print("ERRORS: "+str(counter1) + "/" + str(data_len))
    #print("CAUGHT: "+str(counter2) + "/" + str(counter1))
    #print("TOTAL UNCERTAIN: "+str(dub_cnt) + "/" + str(data_len))
        

def sift_by(trainset, indices, category, minv, maxv):
    return [str(index) for (index,val) in zip(trainset["index"],trainset[category])\
        if ((minv<=float(val) and maxv>=float(val)) and (str(index) in indices))]

def try_to_predict_by(trainset, indices, category, minv, maxv, forcepredict):
    print(len(indices))
    current_list = [int(digit) for (index,digit,val) in zip(trainset["index"],\
        trainset["numbers"], trainset[category])\
        if (str(index) in indices and (minv<=float(val) and maxv>=float(val)))]
        
    freqs_table = collections.Counter(current_list)
    #print(len(freqs_table))
    sorting = sorted(freqs_table, key=freqs_table.get, reverse=True)
    rel_freq = freqs_table[sorting[0]]/float(sum(freqs_table.values()))
    
    if(rel_freq > certainty_cutoff or forcepredict):
        return sorting[0]
    else:
        return -1

def generate_predictions2(trainset, stats):
    
    predictions = []
    stats_len = len(stats)
    
    stats_len = 20
    
    for i in range(0,stats_len):
        indexset = stats["index"]
        #print(len(indexset))
        
        pxl_count = int(stats["pixel_count"][i])
        indexset = sift_by(trainset, indexset, "pixel_count", pxl_count-20,pxl_count+20)
        mass_x = float(stats["mass_x"][i])
        indexset = sift_by(trainset, indexset, "mass_x", mass_x-20,mass_x+20)
        hole_count = int(stats["holes"][i])
        indexset = sift_by(trainset, indexset, "holes", hole_count-1,hole_count+1)
        mass_y = float (stats["mass_y"][i])
        #hole_count = int(stats["holes"][i])
        prediction =\
            try_to_predict_by(trainset, indexset, "mass_y", mass_y-20,mass_y+20, True)
        predictions.append(prediction)

    # ASSUMING TRANSET = STATS
    counter1 = 0
    for i in range(0,stats_len):
        #print("ACTUAL: "+ stats["numbers"][i] + "; PREDICTED: "+\
        #    str(predictions[i])+"; DUBIOUS? "+dubious[i])
        if int(stats["numbers"][i]) != predictions[i]:
            counter1+=1

    print("ERRORS: "+str(counter1) + "/" + str(stats_len))


#data_train = np.array(list(csv.reader(open(stats_export_path+train_file_name,"rb"),\
#    delimiter=',')))

data_test = np.array(list(csv.reader(open(stats_export_path+test_file_name,"rb"),\
    delimiter=',')))


#extract_and_save_statistics(data_train, "Trunc_stats", True)
#extract_and_save_statistics(data_train, "Train_stats", True)
#extract_and_save_statistics(data_test, "Test_stats", False)

train_stats = defaultdict(list)   
with open(stats_export_path+"Train_stats"+".csv") as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value 
            train_stats[k].append(v) # append the value into the appropriate list
                                 # based on column name k
test_stats = defaultdict(list)   
with open(stats_export_path+"Test_stats"+".csv") as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value 
            test_stats[k].append(v) # append the value into the appropriate list
                                 # based on column name k

averages = process_statistics(train_stats)

generate_predictions(averages, test_stats)
#generate_predictions2(stats, stats)

#print(statistics)