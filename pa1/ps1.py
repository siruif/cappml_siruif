# Machine Learning PS1
# CAPP 30254
# Sirui Feng
# siruif@uchicago.edu


from ps1_1 import gen_summary
from ps1_2 import genderize
from ps1_3 import fill_in_missing_values

#gen_summary('mock_student_data.csv','output/summary_stats.txt')
genderize('mock_student_data.csv','output/mock_student_data_genderize.csv')
fill_in_missing_values('output/mock_student_data_genderize.csv')