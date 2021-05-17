#submission.py
#Author: Anant Krishna Mahale
#zID: 5277610


# function to calculate the absolute the difference between the query_hashes and each data_hashes.
def find_absdiff(data_hashes, query_hashes):
    result = []
    for i in range(len(query_hashes)):
        current_result = abs(data_hashes[i]-query_hashes[i]) #finding the absolute difference between data and query hashes.
        result.append(current_result)
    result.sort()  #sorting the result before returning. 
    return result


# function to return the element in the perticular position.
def get_element_from_pos(data_list, position):
    return data_list[position] 

# function to find the offset based on the alpha_m and beta_n values.
def calculate_offset(transformed_data_hashes, alpha_m, beta_n):
    # get the values in the alpha_m position with the indices.
    transformed_data_hashes = transformed_data_hashes.map(lambda data: get_element_from_pos(data[1], alpha_m)).sortBy(
        lambda value: value).zipWithIndex()  # https://stackoverflow.com/questions/36438321/pyspark-rdd-find-index-of-an-element

    # from the alpham_pos_elements get the element which is in beta position.
    transformed_data_hashes = transformed_data_hashes.filter(lambda data: data[1] == beta_n)

    # getting the value from the list.
    offset = transformed_data_hashes.map(lambda data: data[0]).min()  # typically should have only one element
    return offset

# function to check if a perticular key qualifies offset and alpha condition.
def count_function(key_value, transformed_data_list, offset, alpha_m):
    result = 0
    for value in transformed_data_list:
        if value <= offset:
            result = result+1
    if result >= alpha_m:
        return key_value
    return None


########### Question 1 ##########
# given function definition.
def c2lsh(data_hashes, query_hashes, alpha_m, beta_n):
    processed_data = data_hashes.map(
        lambda data: [data[0], find_absdiff(data[1], query_hashes)])
    offset = calculate_offset(processed_data, alpha_m-1, beta_n-1) # -1 is specified as index starts with 0.
    candidates = processed_data.map(lambda data: count_function(
       data[0], data[1], offset, alpha_m)).filter(lambda value: value != None)
    return candidates