import torch
import pickle
import pickletools
import io

def inspect_pth_content(filepath):
    """Inspect .pth file content"""
    print(f"\n=== Content of {filepath} ===")
    try:
        data = torch.load(filepath, map_location='cpu', weights_only=True)
        
        print(f"Type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Number of keys: {len(data)}")
            print("Keys and their types:")
            for key, value in list(data.items())[:5]:  # First 5 items
                print(f"  '{key}': {type(value)}")
                if hasattr(value, 'shape'):
                    print(f"    Shape: {value.shape}")
                    print(f"    Dtype: {value.dtype}")
        else:
            print(f"Content preview: {str(data)[:100]}...")
            
    except Exception as e:
        print(f"Error loading .pth: {e}")
        # Try without weights_only
        try:
            data = torch.load(filepath, map_location='cpu')
            print("Loaded successfully without weights_only=True")
            print(f"Type: {type(data)}")
        except Exception as e2:
            print(f"Still failed: {e2}")

def inspect_pickle_structure(filepath):
    """Inspect pickle file structure without loading it"""
    print(f"\n=== Pickle structure of {filepath} ===")
    try:
        with open(filepath, 'rb') as f:
            # Read first few bytes to see pickle version
            magic = f.read(2)
            print(f"Pickle magic bytes: {magic}")
            
            # Reset and analyze structure
            f.seek(0)
            
            # Capture pickletools output
            output = io.StringIO()
            try:
                pickletools.dis(f, output)
                lines = output.getvalue().split('\n')
                
                print("First 20 lines of pickle disassembly:")
                for i, line in enumerate(lines[:20]):
                    print(f"  {line}")
                    
                print(f"\nTotal lines in pickle disassembly: {len(lines)}")
                
            except Exception as e:
                print(f"Pickletools analysis failed: {e}")
                
    except Exception as e:
        print(f"Error reading pickle file: {e}")

def compare_file_headers(pth_path, pkl_path):
    """Compare the binary headers of both files"""
    print(f"\n=== Binary Header Comparison ===")
    
    try:
        # Read first 100 bytes of each file
        with open(pth_path, 'rb') as f:
            pth_header = f.read(100)
        
        with open(pkl_path, 'rb') as f:
            pkl_header = f.read(100)
        
        print(f".pth header (first 50 bytes): {pth_header[:50]}")
        print(f".pkl header (first 50 bytes): {pkl_header[:50]}")
        
        # Check if they're the same format
        if pth_header[:10] == pkl_header[:10]:
            print("Headers look similar - might be same format with different extension")
        else:
            print("Headers are different - different file formats")
            
    except Exception as e:
        print(f"Error comparing headers: {e}")

# Main execution
if __name__ == "__main__":
    pth_file = "/Users/vatsaldp/Desktop/RA/FiloSegGUI-main/FiloGUI_models/FilaTip_6.pth"  # Update this path
    pkl_file = "/Users/vatsaldp/Desktop/RA/FiloSegGUI-main/pickles/FilaTip_6.pkl"
    
    inspect_pth_content(pth_file)
    inspect_pickle_structure(pkl_file)
    compare_file_headers(pth_file, pkl_file)