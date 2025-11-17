import os
from PIL import Image
import re

def natural_sort_key(s):
    """Sort strings containing numbers in natural order."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def create_collage_gif(input_dir, output_path, fps=10):
    """Create a GIF from three sets of images placed side-by-side."""
    
    # Get all PNG files for each set
    l1_files = sorted([f for f in os.listdir(input_dir) if f.startswith('l1') and f.endswith('.png')], 
                      key=natural_sort_key)
    l2_files = sorted([f for f in os.listdir(input_dir) if f.startswith('l2') and f.endswith('.png')], 
                      key=natural_sort_key)
    l3_files = sorted([f for f in os.listdir(input_dir) if f.startswith('l3') and f.endswith('.png')], 
                      key=natural_sort_key)
    
    # Verify we have matching numbers of images
    num_frames = min(len(l1_files), len(l2_files), len(l3_files))
    if num_frames == 0:
        print("Error: No matching images found!")
        return
    
    print(f"Found {num_frames} frames in each set")
    
    # Create composite frames
    composite_frames = []
    
    for i in range(num_frames):
        # Load images
        img1 = Image.open(os.path.join(input_dir, l1_files[i])).convert('RGBA')
        img2 = Image.open(os.path.join(input_dir, l2_files[i])).convert('RGBA')
        img3 = Image.open(os.path.join(input_dir, l3_files[i])).convert('RGBA')
        
        # Crop all images to top 85%
        crop_height1 = int(img1.height * 0.85)
        img1 = img1.crop((0, 0, img1.width, crop_height1))
        
        crop_height2 = int(img2.height * 0.85)
        img2 = img2.crop((0, 0, img2.width, crop_height2))
        
        crop_height3 = int(img3.height * 0.85)
        img3 = img3.crop((0, 0, img3.width, crop_height3))
        
        # Get dimensions
        width = img1.width + img2.width + img3.width
        height = max(img1.height, img2.height, img3.height)
        
        # Create composite image with white background
        composite = Image.new('RGB', (width, height), (255, 255, 255))
        
        # Paste images with white background (handle transparency)
        white_bg1 = Image.new('RGB', img1.size, (255, 255, 255))
        white_bg1.paste(img1, mask=img1.split()[3] if img1.mode == 'RGBA' else None)
        
        white_bg2 = Image.new('RGB', img2.size, (255, 255, 255))
        white_bg2.paste(img2, mask=img2.split()[3] if img2.mode == 'RGBA' else None)
        
        white_bg3 = Image.new('RGB', img3.size, (255, 255, 255))
        white_bg3.paste(img3, mask=img3.split()[3] if img3.mode == 'RGBA' else None)
        
        composite.paste(white_bg1, (0, 0))
        composite.paste(white_bg2, (img1.width, 0))
        composite.paste(white_bg3, (img1.width + img2.width, 0))
        
        composite_frames.append(composite)
        print(f"Processed frame {i+1}/{num_frames}")
    
    # Save as GIF with maximum quality
    duration = int(1000 / fps)  # milliseconds per frame
    composite_frames[0].save(
        output_path,
        save_all=True,
        append_images=composite_frames[1:],
        duration=duration,
        loop=0,
        optimize=False,
        quality=100
    )
    
    print(f"GIF saved to {output_path}")

if __name__ == "__main__":
    input_directory = "/home/mzomoro1/bin/GO-MELT/gif/2rectangleSlow"
    output_file = "/home/mzomoro1/bin/GO-MELT/gif/2rectangleSlow_collage.gif"
    
    create_collage_gif(input_directory, output_file, fps=10)
