import os
import cv2
import csv
import json

def main():
  # Open image directory
  files_path = "./data_for_test/non_separate_0408"
  files = os.listdir(files_path)

  for fname in files:
    with open("./data_for_test/annotations/annotation.csv","a") as f:
      writer=csv.writer(f)
      path=files_path + '/' + fname
      writer.writerow([path,None,None,None,None,None])

if __name__=='__main__':
  main()
