import sys
from read_explore import go
from fill_missing import fill_in_missing_values


if __name__=="__main__":
    if len(sys.argv) != 2:
        print("usage: python3 {} <raw data filename>".format(sys.argv[0]))
        sys.exit(1)

    input_data = sys.argv[1]
    go(input_data)
    print()
    fill_in_missing_values(input_data)
    print()




    print("Done! Check your output folder.")