import datetime
import os
import re
import shutil


# Function to list all the measurements folders and store in a list
def get_subfolders(folder_path, pattern):
    subfolders = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path) and re.search(pattern, item):
            subfolders.append(item_path)
    return subfolders


# Function to move all the files to the same temporary folder
def move_files(file_type, aux_folder, subfolders):
    for folder in subfolders:
        folder_type = os.path.join(folder, file_type)
        try:
            for file in os.listdir(folder_type):
                file_path = os.path.join(folder_type, file)
                shutil.move(file_path, aux_folder)
                if (
                    not os.listdir(folder_type) or not os.path.exists(folder_type)
                ) and file_type == "dark_current":
                    shutil.rmtree(folder)
        except:  # noqa: E722
            subfolders_check = get_subfolders(folder, r"nt")
            for fold in subfolders_check:
                if not os.listdir(fold) and len(subfolders_check) == 1:
                    shutil.rmtree(folder)


# Function to sort the nightime measurement in the correct way
def correct_nt(files, file_type, aux_folder):
    # Check if this is the dark_current folder in order to compare the ending dc with the last measurement
    months_corr = {
        "1": "01",
        "2": "02",
        "3": "03",
        "4": "04",
        "5": "05",
        "6": "06",
        "7": "07",
        "8": "08",
        "9": "09",
        "A": "10",
        "B": "11",
        "C": "12",
    }
    if file_type == "dark_current":
        is_dc = True
    else:
        is_dc = False

    count = 0
    for file in files:
        folder_check_ms = os.path.join(
            folder_path,
            "20"
            + files[count - 1][1:3]
            + months_corr[files[count - 1][3]]
            + files[count - 1][4:6]
            + "nt",
            "measurements",
        )

        # Creat the first folder and move the first file to this folder
        if count == 0:
            folder_raw = os.path.join(
                folder_path,
                "20" + file[1:3] + str(0) + file[3] + file[4:6] + "nt",
                file_type,
            )
            os.makedirs(folder_raw, exist_ok=True)
            file_path_ms = os.path.join(folder_raw, file)
            shutil.move(os.path.join(aux_folder, file), folder_raw)
            count += 1
            continue

        # Create the datenum object used to compare the measurement time of each file
        datestr_file = (
            "20"
            + file[1:3]
            + "-"
            + months_corr[file[3]]
            + "-"
            + file[4:6]
            + " "
            + file[6:8]
            + ":"
            + file[9:11]
            + ":"
            + file[11:13]
        )
        datestr_file_ant = (
            "20"
            + files[count - 1][1:3]
            + "-"
            + months_corr[files[count - 1][3]]
            + "-"
            + files[count - 1][4:6]
            + " "
            + files[count - 1][6:8]
            + ":"
            + files[count - 1][9:11]
            + ":"
            + files[count - 1][11:13]
        )
        datenum_file = datetime.datetime.strptime(datestr_file, "%Y-%m-%d %H:%M:%S")
        datenum_file_ant = datetime.datetime.strptime(
            datestr_file_ant, "%Y-%m-%d %H:%M:%S"
        )

        # Compare the date of the file with the previous to check if is the same measurement
        if (int(file[4:6]) - int(files[count - 1][4:6])) <= 1 and (
            datenum_file.timestamp() - datenum_file_ant.timestamp() < 6 * 3600
        ):
            file_path_ms = os.path.join(folder_raw, file)
            shutil.move(os.path.join(aux_folder, file), folder_raw)

        # Compare the dark current with the last measurement instead of the last dark current to avoid large interval error between dark currents
        elif os.path.exists(folder_check_ms) and is_dc:
            ms = sorted(os.listdir(folder_check_ms))
            datestr_ms = (
                "20"
                + ms[-1][1:3]
                + "-"
                + months_corr[ms[-1][3]]
                + "-"
                + ms[-1][4:6]
                + " "
                + ms[-1][6:8]
                + ":"
                + ms[-1][9:11]
                + ":"
                + ms[-1][11:13]
            )
            datenum_ms = datetime.datetime.strptime(datestr_ms, "%Y-%m-%d %H:%M:%S")
            # Compare the dark current time with measurement time to check if it is the same nighttime of measurement
            if (
                (int(file[4:6]) - int(ms[-1][4:6])) <= 1
                and (datenum_file.timestamp() - datenum_ms.timestamp() < 6 * 3600)
                and (datenum_file.timestamp() - datenum_ms.timestamp() > 0)
            ):
                file_path_ms = os.path.join(folder_raw, file)
                shutil.move(os.path.join(aux_folder, file), folder_raw)
            else:
                folder_raw = os.path.join(
                    folder_path,
                    "20" + file[1:3] + months_corr[file[3]] + file[4:6] + "nt",
                    file_type,
                )
                os.makedirs(folder_raw, exist_ok=True)
                file_path_ms = os.path.join(folder_raw, file)
                shutil.move(os.path.join(aux_folder, file), folder_raw)
        else:
            folder_raw = os.path.join(
                folder_path,
                "20" + file[1:3] + months_corr[file[3]] + file[4:6] + "nt/",
                file_type,
            )
            os.makedirs(folder_raw, exist_ok=True)
            file_path_ms = os.path.join(folder_raw, file)  # noqa: F841
            shutil.move(os.path.join(aux_folder, file), folder_raw)

        count += 1

    if not os.listdir(aux_folder):
        shutil.rmtree(aux_folder)


PATH = "02-data_raw_organized"
for year in os.listdir(PATH):
    folder_path = os.path.join(PATH, year)
    aux_folder_dc = os.path.join(folder_path, "temporary_folder_dc")
    aux_folder_ms = os.path.join(folder_path, "temporary_folder_ms")
    os.makedirs(aux_folder_dc, exist_ok=True)
    os.makedirs(aux_folder_ms, exist_ok=True)
    subfolders = get_subfolders(folder_path, r"nt")

    file_types = {"measurements": aux_folder_ms, "dark_current": aux_folder_dc}

    for file_type, aux_folder in file_types.items():
        move_files(file_type, aux_folder, subfolders)

    files_dc = sorted(os.listdir(aux_folder_dc))
    files_ms = sorted(os.listdir(aux_folder_ms))

    correct_nt(files_ms, "measurements", aux_folder_ms)
    correct_nt(files_dc, "dark_current", aux_folder_dc)
