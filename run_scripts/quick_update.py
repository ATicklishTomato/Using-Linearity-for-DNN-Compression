

path = "./*/*/*/llama-3-1b/*/*.bash"
target = "--time=08:00:00"
replace = "--time=05:00:00"

if __name__ == "__main__":
    # Find all .bash files in the specified path
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