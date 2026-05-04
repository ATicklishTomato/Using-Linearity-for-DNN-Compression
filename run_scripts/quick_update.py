

path = "./*/procrustes/*/*/*/*.bash"
target = "--mem=16G"
replace = "--mem=32G"

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