import os
import csv
import json

def main():
  # Where is the json file
  f = open("../../Desktop/Orthor_experiment/temp0730/tif_5m/others/vott-json-export/temp_ortho_5m-export.json", 'r')
  json_data = json.load(f)
  image_list = json_data["assets"]
  tag_list = []

  for image in image_list:
    # Where is the dataset directory
    image_name = json_data["assets"][image]["asset"]["name"]
    path = os.path.join("csv_data/temp0730/tif_5m/images/",image_name)

    if not json_data["assets"][image]["regions"]:
      with open("../../Desktop/Orthor_experiment/temp0730/tif_5m/annotations/json_annotation.csv","a") as f:
        writer=csv.writer(f)
        writer.writerow([path,None,None,None,None,None])

    for box in range(len(json_data["assets"][image]["regions"])):
      bboxPath = json_data["assets"][image]["regions"][box]["boundingBox"]
      x1 = round(bboxPath["left"])
      y1 = round(bboxPath["top"])
      x2 = round(bboxPath["left"]+bboxPath["width"])
      y2 = round(bboxPath["top"]+bboxPath["height"])
      tag = json_data["assets"][image]["regions"][box]["tags"][0]

      if not tag_list:
        tag_list.append(tag)
      if my_index_multi(tag_list, tag) == []:
        tag_list.append(tag)
      # Where is the annotation.csv (if you have not made it, it makes it)
      with open("../../Desktop/Orthor_experiment/temp0730/tif_5m/annotations/json_annotation.csv","a") as f:
        writer=csv.writer(f)
        writer.writerow([path,x1,y1,x2,y2,tag])

  for i in range(len(tag_list)):
    tag = tag_list[i]
    # Where is the test_class_id.csv (if you have not made it, it make it)
    with open("../../Desktop/Orthor_experiment/temp0730/tif_5m/annotations/json_class_id.csv","a") as f:
      writer=csv.writer(f)
      writer.writerow([tag,i])

def my_index_multi(l, x):
    return [i for i, _x in enumerate(l) if _x == x]

if __name__=='__main__':
  main()
