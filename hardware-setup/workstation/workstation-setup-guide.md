# Digital Twin Workstation Setup Guide

This guide provides detailed instructions for setting up a Digital Twin workstation for the Physical AI & Humanoid Robotics curriculum.

## System Requirements

### Minimum Specifications
- **CPU**: Intel i7-12700K or AMD Ryzen 7 5800X
- **GPU**: NVIDIA RTX 4070 (12GB VRAM)
- **RAM**: 32GB DDR4-3200
- **Storage**: 1TB NVMe SSD
- **OS**: Ubuntu 22.04 LTS

### Recommended Specifications
- **CPU**: Intel i9-13900K or AMD Ryzen 9 7950X
- **GPU**: NVIDIA RTX 4070 Ti+ (16GB+ VRAM) or RTX 4080/4090
- **RAM**: 64GB DDR4-3200 or DDR5-5200
- **Storage**: 2TB+ NVMe SSD
- **OS**: Ubuntu 22.04 LTS

## Installation Steps

### 1. Operating System
Install Ubuntu 22.04 LTS from the official website. During installation:
- Select "Normal installation" with third-party drivers
- Ensure system is connected to internet for additional driver installation

### 2. System Updates
```bash
sudo apt update && sudo apt upgrade -y
```

### 3. NVIDIA GPU Drivers
```bash
# Check recommended driver
ubuntu-drivers devices

# Install recommended driver
sudo ubuntu-drivers autoinstall

# Reboot to apply changes
sudo reboot
```

### 4. CUDA Toolkit
```bash
# Download CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run

# Run installer
sudo sh cuda_12.1.0_530.30.02_linux.run
```

### 5. Essential Development Tools
```bash
sudo apt install build-essential cmake git curl wget python3-dev python3-pip python3-venv
```

### 6. Graphics and Display
```bash
# Install graphics drivers and libraries
sudo apt install mesa-utils libgl1-mesa-glx libgl1-mesa-dri
```

## Verification

### GPU Verification
```bash
nvidia-smi
nvidia-ml-py3
```

### CUDA Verification
```bash
nvcc --version
nvidia-ml-py3
```

## Performance Tuning

### 1. Power Management
```bash
# To prevent CPU throttling during intensive tasks
sudo apt install tlp tlp-rdw
sudo tlp start
```

### 2. Storage Optimization
- Ensure SSD is used for all development work
- Place ROS workspace on SSD for faster build times
- Use separate drive for simulation assets if available

### 3. Memory Management
- Consider 64GB RAM for complex simulations
- Monitor memory usage during heavy simulation tasks
- Close unnecessary applications when running simulations

## Troubleshooting

### Common Issues:

1. **Display problems after driver installation**:
   - Boot into recovery mode and reinstall drivers
   - Ensure secure boot is disabled in UEFI settings

2. **CUDA not working**:
   - Check that the installed CUDA version matches your GPU
   - Verify PATH and LD_LIBRARY_PATH include CUDA paths

3. **Simulation running slowly**:
   - Check GPU drivers and ensure CUDA is working
   - Verify sufficient VRAM availability
   - Close other GPU-intensive applications

## Next Steps

After completing the workstation setup, proceed with:
1. ROS 2 installation
2. Gazebo Harmonic setup
3. NVIDIA Isaac Sim installation
4. Unity installation for simulation

## Additional Resources

- [Ubuntu Installation Guide](https://ubuntu.com/tutorials/install-ubuntu-desktop)
- [NVIDIA Driver Installation](https://www.nvidia.com/drivers/)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [System Performance Tuning](https://wiki.ubuntu.com/Kernel/PowerManagement/Logind)