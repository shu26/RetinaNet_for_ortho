import os
import cv2
import csv
import json

def main():
  # Open image directory
  files_path = "../../Desktop/Orthor_experiment/test_dataset/komesu/images"
  files = os.listdir(files_path)

  for fname in files:
    with open("test_dataset/komesu/annotations/annotation.csv","a") as f:
      writer=csv.writer(f)
      path=files_path + '/' + fname
      writer.writerow([path,None,None,None,None,None])

if __name__=='__main__':
  main()
