import pandas


luna_csv_path = "/content/"

positive_nodules = pandas.read_csv(luna_csv_path + "annotations.csv")
patient_names = pandas.read_csv(luna_csv_path + "names.csv")

#positive_nodules.head()

#patient_names.head()


for i in range(len(patient_names)):
  if patient_names.at[i, 'Patient_ID'] in positive_nodules.values:
    patient_names.at[i,'status'] = 1
    print (patient_names.at[i, 'Patient_ID'] ,i, ' is positive!')
  else:
    patient_names.at[i,'status'] = 0
    print (patient_names.at[i, 'Patient_ID'] , i, ' is negative!')
patient_names.to_csv('truth_table.csv', index=False)

#patient_names.head()