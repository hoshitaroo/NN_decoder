import csv
import numpy as np

from param import param
from toric_code import ToricCode

toric_code = ToricCode()

def get_error():
    return toric_code.generate_errors

def get_syndrome_x(errors):
    return toric_code.generate_syndrome_X(errors)

def get_syndrome_z(errors):
    return toric_code.generate_syndrome_Z(errors)


num_of_data = 200

csv_file_data = open('test_data.csv', 'w', newline='', encoding='utf-8')
csv_file_label = open('test_label.csv', 'w', newline='', encoding='utf-8')

csv_data_writer = csv.writer(csv_file_data)
csv_label_writer = csv.writer(csv_file_label)

for i in range(num_of_data):
    errors = toric_code.generate_errors()
    label = errors.flatten()
    syndrome_x = toric_code.generate_syndrome_X(errors)
    syndrome_z = toric_code.generate_syndrome_Z(errors)
    data = np.concatenate((syndrome_x.flatten(), syndrome_z.flatten()))
    csv_data_writer.writerow(data)
    csv_label_writer.writerow(label)


csv_file_data.close()
csv_file_label.close()