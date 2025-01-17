import glfw
import OpenGL.GL as gl
import numpy as np
import imageio
import util
import util_gau
import imageio
import os
import argparse
from renderer_ogl import OpenGLRenderer, GaussianRenderBase
from scipy.spatial.transform import Rotation as R

# Camera rotation angle increment (10 degrees)
ROTATION_STEP = 10
# Number of images (full 360 degrees)
NUM_IMAGES = 360 // ROTATION_STEP
# Distance of the camera from the object (adjust to fit the vehicle's size)
CAMERA_DISTANCE = 100.0
# Center of the vehicle (adjust if the object is off-center)
VEHICLE_CENTER = np.array([-4,-5,5])
CAMERA_ORIGINAL_POS = np.array([-4,-9,5])#back,down,right
FOV = 0.2

def impl_glfw_init(window_name, width, height):
    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(width, height, window_name, None, None)
    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    glfw.make_context_current(window)
    glfw.swap_interval(1)
    gl.glViewport(0, 0, width, height)
    return window

INCLINATION_AMPLITUDE = 1.0

def render_and_save_image(renderer, camera, window, mode, angle, output_name, depth_pass=False):
    width, height = camera.w, camera.h
    camera.fovy = FOV 
    camera.is_intrin_dirty = True  # Mark intrinsics as dirty, so they will be updated
    renderer.update_camera_intrin(camera)
    
    # Compute the camera position in a circular orbit around the object
    cam_x = CAMERA_DISTANCE * np.cos(np.radians(angle))
    cam_z = CAMERA_DISTANCE * np.sin(np.radians(angle))
    cam_y = -INCLINATION_AMPLITUDE * np.cos(np.radians(angle))  # Inclination component

    camera.position = np.array([cam_x+CAMERA_ORIGINAL_POS[0], cam_y+CAMERA_ORIGINAL_POS[1], cam_z+CAMERA_ORIGINAL_POS[2]])  # Adjust y position accordingly
    camera.target = VEHICLE_CENTER  # Aim the camera at the center of the vehicle
    
    # Calculate the tilt angle based on the inclination
    tilt_angle = np.radians(INCLINATION_AMPLITUDE * np.cos(np.radians(angle)) / CAMERA_DISTANCE)
    
    # Create a rotation matrix for the tilt around the -z axis
    tilt_rotation = R.from_euler('x', tilt_angle).as_matrix()
    
    # Apply the tilt to the camera's up vector
    camera.up = tilt_rotation @ np.array([0, -1, 0])
    
    camera.is_pose_dirty = True
    renderer.sort_and_update(camera)
    # Update camera pose
    renderer.update_camera_pose(camera)

    # Clear the screen (both color and depth buffer)
    

    # Set render mode
    if depth_pass:
        gl.glClearColor(0, 0, 0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        # Activate depth rendering mode programmatically
        g_render_mode = 3  # Assuming this corresponds to depth based on your combo box
        renderer.set_render_mod(g_render_mode - 4)  # Adjust index if needed
    else:
        gl.glClearColor(1, 1, 1, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        renderer.set_render_mod(mode)

    # Render the scene
    renderer.draw()
    print(f"Rendering {width} x {height} image at angle {angle} degrees")
    # Read color buffer for color pass
    bufferdata = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    img = np.frombuffer(bufferdata, np.uint8).reshape(height, width, 3)
    imageio.imwrite(output_name, img[::-1])
    if depth_pass:
        auto_level_adjustment(output_name)
def auto_level_adjustment(image_path):
    # Read the image
    img = imageio.imread(image_path)
    
    # Convert to grayscale if it's not already
    if len(img.shape) == 3:
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    
    # Find the minimum and maximum pixel values
    min_val = np.min(img)
    max_val = np.max(img)
    
    # Normalize the pixel values to the range [0, 255]
    img_normalized = (img - min_val) / (max_val - min_val) * 255
    img_normalized = img_normalized.astype(np.uint8)
    
        # Save the adjusted image
    imageio.imwrite(image_path, img_normalized)

def main():
    parser = argparse.ArgumentParser(description="Render 36 images around a Gaussian Splatting object from a .ply file.")
    parser.add_argument("--ply", required=True, help="Path to the .ply file.")
    args = parser.parse_args()

    # Initialize camera and renderer
    camera = util.Camera(1024, 2048)
    window = impl_glfw_init("Gaussian Splat Viewer", camera.w, camera.h)

    # Init renderer
    renderer = OpenGLRenderer(camera.w, camera.h)
    
    # Load Gaussian data from the provided .ply file
    gaussians = util_gau.load_ply(args.ply)
    
    # Update Gaussian data in the renderer
    renderer.update_gaussian_data(gaussians)
    
    # Initial render setup to ensure everything is initialized correctly
    renderer.sort_and_update(camera)
    renderer.set_scale_modifier(1.0)
    renderer.update_camera_pose(camera)
    renderer.update_camera_intrin(camera)
    
    # Save directory
    save_dir = "rendered_images"
    os.makedirs(save_dir, exist_ok=True)

    # Render frames at different angles
    for i in range(NUM_IMAGES):
        angle = i * ROTATION_STEP

        # Render color image
        color_img_name = f"{save_dir}/color_{angle:03d}.png"
        render_and_save_image(renderer, camera, window, mode=3, angle=angle, output_name=color_img_name)

        # Render depth image
        depth_img_name = f"{save_dir}/depth_{angle:03d}.png"
        render_and_save_image(renderer, camera, window, mode=4, angle=angle, output_name=depth_img_name, depth_pass=True)

    glfw.terminate()

if __name__ == "__main__":
    main()
