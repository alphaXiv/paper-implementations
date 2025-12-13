import subprocess
import time
import os
import tqdm 
import shutil

# Configuration
N = 10  # Number of times to run the command
files_list = os.listdir('images_to_pdf')
dir_ = len(files_list) 
# input_pdf = "images_to_pdf/book_en_A.Concise.Introduction.to.Linear.Algebra,.Geza.Schay,.Birkhauser,.2012_page_194.pdf"
output_dir = "./output"
method = "hf"

# Run the command N times
for i in tqdm.tqdm(range(dir_)):
    print(f"\n{'='*60}")
    print(f"Running iteration {i+1}/{dir_}")
    print(f"{'='*60}")

    input_pdf = os.path.join('images_to_pdf', files_list[i])
    cmd = [
        "chandra",
        input_pdf,
        output_dir,
        "--method",
        method
    ]
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed_time = time.time() - start_time
        
        print(f"Command: {' '.join(cmd)}")
        print(f"Exit code: {result.returncode}")
        print(f"Time taken: {elapsed_time:.2f} seconds")
        
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
            
        if result.returncode != 0:
            print(f"Warning: Command failed with exit code {result.returncode}")

        # Get the filename without extension for the output path
        filename = os.path.splitext(files_list[i])[0]
        md_path = os.path.join('output', filename, filename + '.md')
        output_path = os.path.join('outputs_final')
        shutil.move(md_path, output_path)
    except Exception as e:
        print(f"Error running command: {e}")
    
    # Optional: add a small delay between iterations
    if i < N - 1:  # Don't sleep after the last iteration
        time.sleep(1)

print(f"\n{'='*60}")
print(f"Completed all {N} iterations")
print(f"{'='*60}")