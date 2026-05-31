

path = "./rq2/hybridization/*/resnet34/*/*.bash"
target = '-e relation'
replace = '-e hybridization -l fraction --threshold "25%"'

if __name__ == "__main__":
    """This script updates all .bash files in the specified path by replacing the target string with the new string.
    It searches for the target string in each file, and if found, it replaces it and saves the updated content back to the file. 
    It also prints out which files were updated and which did not contain the target string.
    """
    import glob
    files = glob.glob(path, recursive=True)
    for file in files:
        with open(file, 'r') as f:
            content = f.read()
        if target in content:
            new_content = content.replace(target, replace)
            with open(file, 'w') as f:
                f.write(new_content)
            print(f"Updated: {file}")
        else:
            print(f"No match found in: {file}")