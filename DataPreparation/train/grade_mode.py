def grade_mode(list):
    '''
    calculate mode

    parameter：
        type:list

    return：
        type: list

    '''

    list_set = set(list)  # del repeat element
    frequency_dict = {}
    for i in list_set:  # make element:count dict 
        frequency_dict[i] = list.count(i)  
    grade_mode = []
    sorted_frequency_dict = sorted(frequency_dict.items(), key=lambda item: item[1], reverse=True) #sort dict by count in reverse order.
    sorted_frequency_dict = dict(sorted_frequency_dict[:5])
    for i, (key, value) in enumerate(sorted_frequency_dict.items()):
        if i >= 3 and value <= 10:
            continue #at most 3st grade mode, count must great than 10
        grade_mode.append(key)
        print('max match mode {key}:{value}'.format(key=key, value=value))
    return grade_mode

if __name__ == '__main__':
    grade_list = [100, 98, 87, 65, 82, 99, 92, 99, 100]
    print(grade_mode(grade_list))