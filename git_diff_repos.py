import os
import difflib
import argparse
from pathlib import Path

def get_file_list(base_dir, subdir=""):
    """Get list of all files in directory/subdirectory, excluding .git directory."""
    base_path = os.path.join(base_dir, subdir) if subdir else base_dir
    if not os.path.exists(base_path):
        raise ValueError(f"Directory not found: {base_path}")
    
    files = []
    for root, _, filenames in os.walk(base_path):
        if '.git' in root:
            continue
        for filename in filenames:
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, base_path)
            files.append(rel_path)
    return set(files)

def compare_dirs(dir1, dir1_subdir, dir2, dir2_subdir, output_dir="diff_output"):
    """Compare two local directories."""
    try:
        # Get file lists
        files1 = get_file_list(dir1, dir1_subdir)
        files2 = get_file_list(dir2, dir2_subdir)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Compare files
        common_files = files1.intersection(files2)
        only_in_dir1 = files1 - files2
        only_in_dir2 = files2 - files1
        
        # Generate summary report
        with open(os.path.join(output_dir, "comparison_summary.txt"), "w") as f:
            f.write("Directory Comparison Summary\n")
            f.write("===========================\n\n")
            f.write(f"Files only in {os.path.join(dir1, dir1_subdir)}:\n")
            for file in sorted(only_in_dir1):
                f.write(f"  {file}\n")
            f.write(f"\nFiles only in {os.path.join(dir2, dir2_subdir)}:\n")
            for file in sorted(only_in_dir2):
                f.write(f"  {file}\n")
            f.write(f"\nCommon files: {len(common_files)}\n")
            
            # Count files with differences
            diff_count = 0
        
        # Generate diffs for common files
        for file in common_files:
            file1_path = os.path.join(dir1, dir1_subdir, file)
            file2_path = os.path.join(dir2, dir2_subdir, file)
            
            try:
                with open(file1_path, 'r', encoding='utf-8') as f1, \
                     open(file2_path, 'r', encoding='utf-8') as f2:
                    diff = difflib.unified_diff(
                        f1.readlines(),
                        f2.readlines(),
                        fromfile=f"dir1/{file}",
                        tofile=f"dir2/{file}"
                    )
                    
                    diff_content = ''.join(diff)
                    if diff_content:
                        diff_count += 1
                        diff_file = os.path.join(output_dir, f"{file.replace('/', '_')}.diff")
                        os.makedirs(os.path.dirname(diff_file), exist_ok=True)
                        with open(diff_file, 'w', encoding='utf-8') as f:
                            f.write(diff_content)
            except UnicodeDecodeError:
                print(f"Warning: Skipping binary file {file}")
        
        # Append diff count to summary
        with open(os.path.join(output_dir, "comparison_summary.txt"), "a") as f:
            f.write(f"\nFiles with differences: {diff_count}\n")
        
        print(f"\nComparison complete. Results saved in {output_dir}/")
        print(f"Found {diff_count} files with differences")
        
    except Exception as e:
        print(f"Error during comparison: {str(e)}")

if __name__ == "__main__":
    '''
    python git_diff_repos.py /mnt/nas/suehyun/alignment-handbook . /mnt/nas/suehyun/BARC finetune/alignment-handbook
    '''
    parser = argparse.ArgumentParser(description="Compare two local directories")
    parser.add_argument("dir1", help="Path to first directory")
    parser.add_argument("dir1_subdir", help="Subdirectory in first directory (or '.' for root)")
    parser.add_argument("dir2", help="Path to second directory")
    parser.add_argument("dir2_subdir", help="Subdirectory in second directory (or '.' for root)")
    parser.add_argument("--output-dir", default="diff_output", help="Output directory for diff files")
    
    args = parser.parse_args()
    compare_dirs(args.dir1, args.dir1_subdir, args.dir2, args.dir2_subdir, args.output_dir)