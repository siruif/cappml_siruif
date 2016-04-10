# Machine Learning PS2 | ML Pipeline
# CAPP 30254
# Sirui Feng
# siruif@uchicago.edu

'''
*** Note: Please have a folder called output in your current directory to pick up all
files that will be generated. And a folder called charts in output folder to colect
graphs.
'''

import sys
import pandas as pd
from read_explore import read_explore
from fill_missing import fill_in_missing_values
from features import generate_features
from build_classifier import build_classifier


if __name__=="__main__":
    if len(sys.argv) != 2:
        print("usage: python3 {} <raw data filename>".format(sys.argv[0]))
        sys.exit(1)

    input_data = sys.argv[1]
    df = read_explore(input_data)
    print()
    df = fill_in_missing_values(df)
    print()
    df_clean = generate_features(df)
    print()
    build_classifier(df_clean)





    print("Done! Check your output folder.")