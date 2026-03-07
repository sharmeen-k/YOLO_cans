import cv2
import numpy as np
import random

class BlurAugment:
    """
    Handles synthetic blur generation to simulate motion and vibration
    artifacts observed on the Viam Rover.
    """
    def __init__(self, blur_prob=0.5, min_kernel=3, max_kernel=15):
        self.blur_prob = blur_prob
        self.min_kernel = min_kernel
        self.max_kernel = max_kernel

    def apply_motion_blur(self, image, kernel_size=None, angle=None):
        """
        Applies motion blur to an image.
        """
        if kernel_size is None:
            kernel_size = random.randint(self.min_kernel, self.max_kernel)
        
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        if angle is None:
            angle = random.randint(0, 360)

        # Create motion blur kernel
        M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(kernel_size))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (kernel_size, kernel_size))
        motion_blur_kernel = motion_blur_kernel / kernel_size

        # Apply filter
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)
        return blurred

    def apply_gaussian_blur(self, image, kernel_size=None):
        """
        Applies Gaussian blur to an image (simulating out-of-focus).
        """
        if kernel_size is None:
            kernel_size = random.randint(self.min_kernel, self.max_kernel)
            
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def __call__(self, image):
        """
        Apply random blur to the image with probability self.blur_prob.
        Expects image in BGR (OpenCV) or RGB format.
        """
        if random.random() > self.blur_prob:
            return image

        # 50/50 chance of motion blur vs gaussian blur (or simulate mixed vibration)
        if random.random() < 0.6:
            # Motion blur is more likely for a moving rover
            return self.apply_motion_blur(image)
        else:
            return self.apply_gaussian_blur(image)

if __name__ == "__main__":
    # Test execution
    print("Testing BlurAugment...")
    aug = BlurAugment(blur_prob=1.0)
    
    # Create a dummy image (checkerboard)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[::20, ::20] = 255
    
    blurred = aug(img)
    cv2.imwrite("project/debug_samples/test_blur_output.jpg", blurred)
    print("Saved test output to project/debug_samples/test_blur_output.jpg")

class BatchedBlurAugment:
    """
    PyTorch Batched GPU implementation of BlurAugment to optimize training speeds.
    Operates entirely on (B, C, H, W) tensors.
    """
    def __init__(self, blur_prob=0.8, min_kernel=3, max_kernel=15, device='cpu'):
        self.blur_prob = blur_prob
        self.min_kernel = min_kernel
        self.max_kernel = max_kernel
        self.device = device
        
    def _get_motion_blur_kernel(self, kernel_size, angle):
        M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(kernel_size))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (kernel_size, kernel_size))
        motion_blur_kernel = motion_blur_kernel / np.sum(motion_blur_kernel)
        import torch
        return torch.from_numpy(motion_blur_kernel).float()

    def _get_gaussian_blur_kernel(self, kernel_size):
        import torch
        # 1D gaussian
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        # 2D gaussian
        gauss_2d = gauss_1d[:, None] @ gauss_1d[None, :]
        return gauss_2d

    def __call__(self, images_tensor):
        """
        images_tensor: (B, C, H, W) on self.device
        Applies a batched GPU-accelerated convolution without Python loops.
        """
        import torch
        import torch.nn.functional as F
        B, C, H, W = images_tensor.shape
        blurred_images = images_tensor.clone()
        
        # Determine which images get blurred
        blur_mask = torch.rand(B, device=self.device) < self.blur_prob
        
        if not blur_mask.any():
            return blurred_images
            
        # We will build a batch of specific kernels for each image that needs blurring
        num_to_blur = blur_mask.sum().item()
        blur_indices = blur_mask.nonzero(as_tuple=True)[0]
        
        # Default kernel size for padding logic across the batch
        # To batch conv2d with different kernels, they must be the same spatial size.
        # We will pad smaller kernels to max_kernel size.
        max_k = self.max_kernel
        if max_k % 2 == 0: max_k += 1
        
        # Create a batched kernel tensor: [num_to_blur * C, 1, max_k, max_k]
        kernels = torch.zeros(num_to_blur * C, 1, max_k, max_k, device=self.device)
        
        for idx in range(num_to_blur):
            kernel_size = random.randint(self.min_kernel, self.max_kernel)
            if kernel_size % 2 == 0: kernel_size += 1
            
            if random.random() < 0.6:
                angle = random.randint(0, 360)
                k = self._get_motion_blur_kernel(kernel_size, angle).to(self.device)
            else:
                k = self._get_gaussian_blur_kernel(kernel_size).to(self.device)
            
            # Pad kernel to max_k
            pad_size = (max_k - kernel_size) // 2
            if pad_size > 0:
                k = F.pad(k, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
            
            # Broadcast to C channels 
            # k is [max_k, max_k], we need it to be copied C times in the batch dimension
            for c in range(C):
                kernels[idx * C + c, 0] = k
                
        # Get the images to blur: [num_to_blur, C, H, W]
        imgs_to_blur = images_tensor[blur_indices]
        
        # Reshape for grouped convolution: [1, num_to_blur * C, H, W]
        imgs_reshaped = imgs_to_blur.view(1, num_to_blur * C, H, W)
        
        # Apply grouped convolution
        padding = max_k // 2
        blurred_reshaped = F.conv2d(imgs_reshaped, kernels, padding=padding, groups=num_to_blur * C)
        
        # Reshape back to [num_to_blur, C, H, W]
        blurred_out = blurred_reshaped.view(num_to_blur, C, H, W)
        
        # Put back into the batch, casting to original dtype if needed (e.g. AMP float16)
        blurred_images[blur_indices] = blurred_out.to(blurred_images.dtype)
            
        return blurred_images

