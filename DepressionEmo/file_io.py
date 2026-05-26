import csv
import sys
import time as custom_time
import json
from numpyencoder import NumpyEncoder

maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


def write_data_to_csv_file(file_name, header_list, data_dict):
    try:
        file = open(file_name, 'a', newline = '', encoding = 'utf-8')
        with file:
            writer = csv.DictWriter(file, fieldnames = header_list)
            writer.writerow(data_dict)
    except Exception as e:
        print('Error -- write_data_to_csv_file: ', e)
        pass

        
def write_to_text_file(file_name, data):
    try:
        with open(file_name, 'a', encoding='utf-8') as f:
            f.write(data + '\n')
        f.close()
    except Exception as e:
        print('Error -- write_to_text_file: ', e)
    
	
def write_to_new_text_file(file_name, data):
    try:
        with open(file_name, 'w', encoding='utf-8') as f:
            if (data == ''):
                f.write(data)
            else:
                f.write(data + '\n')
        f.close()
    except Exception as e:
        print('Error -- write_to_new_text_file: ', e)


def write_list_to_json_file(file_name, data_list, file_access = 'a'):
    try:
        with open(file_name, file_access, encoding='utf-8') as outfile:
            json.dump(data_list, outfile, indent=4, separators=(', ', ': '), ensure_ascii=False, cls=NumpyEncoder)
        outfile.close()
    except Exception as e:
        print('Error -- write_list_to_json_file: ', e)

def write_list_to_jsonl_file(file_name, data_list, file_access = 'a'):
    try:
        with open(file_name, file_access, encoding='utf-8') as outfile:
            for item in data_list:
                json.dump(item, outfile, separators=(', ', ': '), ensure_ascii=False, cls=NumpyEncoder)
                outfile.write('\n')
        outfile.close()
    except Exception as e:
        print('Error -- write_list_to_jsonl_file: ', e)

def read_list_from_json_file(out_file_name, format_json = True, try_no = 0):
    result_list = []
    try:
        with open(out_file_name, 'r', encoding='utf-8') as outfile:
            text = outfile.read()
            text = text.strip(',\n')
            if (format_json == False):
                result_list = json.loads('[' + text + ']') 
            else:
                result_list = json.loads(text)
        outfile.close()
    except Exception as e:
        print('Error -- read_list_from_json_file: ', e)
        try_no += 1
        if (try_no <= 10):
            custom_time.sleep(2)
            return read_list_from_json_file(out_file_name, format_json, try_no)
        pass
    
    return result_list

def read_list_from_jsonl_file(out_file_name, try_no = 0):
    result_list = []
    i = 0
    try:
        with open(out_file_name, 'r', encoding='utf-8') as outfile:
            for line in outfile:
                item = json.loads(line) 
                result_list.append(item)
                i += 1
        outfile.close()
    except Exception as e:
        print('Error -- read_list_from_jsonl_file: ', e, '-- check line: ', i)
        try_no += 1
        if (try_no <= 10):
            custom_time.sleep(2)
            return read_list_from_jsonl_file(out_file_name, try_no)
        pass
    
    return result_list

def read_list_from_text_file(file_name):
    page_list = []
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
              page_list.append(line.strip())
        f.close()
    except:
        print('Error -- read_list_from_text_file')
        with open(file_name, 'a', encoding='utf-8') as f:
            f.close()

    if (len(page_list) == 1):
        return page_list[0]
    
    return page_list

def read_from_text_file(file_name):
    data = ''
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = f.read()
        f.close()
    except:
        pass
    return data
