import json


def extract_json_from_script(all_scripts, start_split, end_split, trim_length=None):
    parts = []
    for script in all_scripts:
        try:
            scriptstr = str(script)
            parts = scriptstr.split(start_split)
        except ValueError as e:
            print(e)
            pass
        if len(parts) > 1:
            break
    jsonstr = parts[1].split(end_split)[0].strip()
    if trim_length is not None:
        jsonstr = jsonstr[:trim_length]
    jsondict = json.loads(jsonstr)
    return jsondict
