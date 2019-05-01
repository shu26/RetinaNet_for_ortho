import os
import csv
import json

def main():
  # Where is the json file
  f = open("anchi/images.json", 'r')
  json_data = json.load(f)
  image_list = json_data["visitedFrames"]
  tag_list = []

  for image in image_list:
    # Where is the dataset directory
    path = os.path.join("./anchi/images",image)

    if not json_data["frames"][image]:
      with open("./anchi/annotations/annotation.csv","a") as f:
        writer=csv.writer(f)
        writer.writerow([path,None,None,None,None,None])

    for box in range(len(json_data["frames"][image])):
      x1 = round(json_data["frames"][image][box]["box"]["x1"])
      y1 = round(json_data["frames"][image][box]["box"]["y1"])
      x2 = round(json_data["frames"][image][box]["box"]["x2"])
      y2 = round(json_data["frames"][image][box]["box"]["y2"])
      tag = json_data["frames"][image][box]["tags"][0]

      if not tag_list:
        tag_list.append(tag)
      if my_index_multi(tag_list, tag) == []:
        tag_list.append(tag)
      # Where is the annotation.csv (if you have not made it, it makes it)
      with open("./anchi/annotations/annotation.csv","a") as f:
        writer=csv.writer(f)
        writer.writerow([path,x1,y1,x2,y2,tag])

  for i in range(len(tag_list)):
    tag = tag_list[i]
    # Where is the test_class_id.csv (if you have not made it, it make it)
    with open("./anchi/annotations/class_id.csv","a") as f:
      writer=csv.writer(f)
      writer.writerow([tag,i])

def my_index_multi(l, x):
    return [i for i, _x in enumerate(l) if _x == x]

if __name__=='__main__':
  main()
