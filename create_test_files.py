"""Create test files for the multimodal claims investigator."""
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Create test_files directory
os.makedirs('test_files', exist_ok=True)

# 1. Create test image (damage photo)
print("Creating test image...")
img = Image.new('RGB', (800, 600), color='white')
draw = ImageDraw.Draw(img)
draw.rectangle([100, 100, 700, 500], outline='red', width=5)
draw.text((150, 150), "Vehicle Damage Photo", fill='black')
draw.text((150, 200), "Front bumper collision", fill='black')
draw.text((150, 250), "Claim: TEST-001", fill='black')
img.save('test_files/damage_photo.jpg')
print("✓ Created: test_files/damage_photo.jpg")

# 2. Create test PDF (simple text file saved as PDF-like)
print("Creating test PDF...")
pdf_content = """TEST INSURANCE CLAIM REPORT

Claim ID: TEST-001
Date: January 15, 2024
Claimant: John Doe
Policy Number: POL-12345

INCIDENT DESCRIPTION:
Vehicle collision at intersection of Main St and Oak Ave.
Front bumper damage to insured vehicle.
Other party ran red light.

ESTIMATED DAMAGES:
- Front bumper replacement: $1,200
- Paint work: $800
- Total: $2,000

WITNESS STATEMENTS:
Witness confirmed other vehicle ran red light.

ADJUSTER NOTES:
Claim appears valid. Recommend approval.
"""
with open('test_files/claim_report.txt', 'w') as f:
    f.write(pdf_content)
print("✓ Created: test_files/claim_report.txt (use as PDF substitute)")

# 3. Create test audio file (WAV format - simple tone)
print("Creating test audio...")
try:
    import wave
    import struct
    
    sample_rate = 44100
    duration = 3  # seconds
    frequency = 440  # A4 note
    
    with wave.open('test_files/driver_statement.wav', 'w') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        for i in range(int(sample_rate * duration)):
            value = int(32767 * 0.3 * np.sin(2 * np.pi * frequency * i / sample_rate))
            wav_file.writeframes(struct.pack('h', value))
    
    print("✓ Created: test_files/driver_statement.wav")
except Exception as e:
    print(f"✗ Audio creation failed: {e}")

# 4. Create test video (MP4 - requires opencv)
print("Creating test video...")
try:
    import cv2
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('test_files/dashcam_footage.mp4', fourcc, 20.0, (640, 480))
    
    for i in range(60):  # 3 seconds at 20 fps
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark gray background
        
        # Add text
        cv2.putText(frame, 'DASHCAM FOOTAGE', (150, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'Frame: {i}/60', (200, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, 'Claim: TEST-001', (180, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add moving rectangle (simulating vehicle)
        x = int(50 + i * 8)
        cv2.rectangle(frame, (x, 350), (x + 100, 420), (0, 0, 255), -1)
        
        video.write(frame)
    
    video.release()
    print("✓ Created: test_files/dashcam_footage.mp4")
except Exception as e:
    print(f"✗ Video creation failed: {e}")
    print("  Install opencv-python: pip install opencv-python")

print("\n=== Test Files Created ===")
print("Location: test_files/")
print("\nYou can now upload these files to test the system:")
print("  - damage_photo.jpg (image)")
print("  - claim_report.txt (text document)")
print("  - driver_statement.wav (audio)")
print("  - dashcam_footage.mp4 (video)")
