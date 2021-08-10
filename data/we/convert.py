import json
def transform(file):
    with open(file) as data:
        data = json.load(data)
    new_data = []
    for student in data:
        new_data.append({})
        new_data[-1]["marks"] = {}
        new_data[-1]["marks"]["ev1"] = student["marks"].get("ev1", None)
        new_data[-1]["marks"]["ev2"] = student["marks"].get("ev2", None)
        new_data[-1]["marks"]["ex1"] = student["marks"].get("ex1", None)
        new_data[-1]["series"] = []
        for series in student["series"]:
            new_data[-1]["series"].append({})
            new_data[-1]["series"][-1]["deadline"] = series["deadline"]
            new_data[-1]["series"][-1]["exercises"] = []
            for exercises in series["exercises"]:
                new_data[-1]["series"][-1]["exercises"].append([])
                for submission in exercises:
                    new_data[-1]["series"][-1]["exercises"][-1].append({})
                    new_data[-1]["series"][-1]["exercises"][-1][-1]["status"] = submission["status"]
                    new_data[-1]["series"][-1]["exercises"][-1][-1]["time"] = submission["time"]
    with open(file, 'w') as data:
        json.dump(new_data, data)

transform("studentdata1617.json")
transform("studentdata1718.json")
transform("studentdata1819.json")
